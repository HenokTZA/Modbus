#!/usr/bin/env python3
import asyncio, inspect, time, json, os, contextlib
from typing import Dict, Tuple, Any, List

from pymodbus.datastore import (
    ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock
)
from pymodbus.framer import FramerType
    # note: FramerType.RTU works for both client & server
from pymodbus.server import StartAsyncSerialServer, StartAsyncTcpServer
from pymodbus.client import AsyncModbusSerialClient

from fastapi import FastAPI, Body, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn

from alarms import AlarmsEngine
from pathlib import Path
from contextlib import asynccontextmanager

# ===================== NEW: Auth & DB imports =====================
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.hash import argon2
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, select
# ================================================================

app = FastAPI()

# serve ./static at /static (use absolute path so it works no matter the cwd)
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

ALARM_ENGINE = AlarmsEngine()
LAST_GOOD_POLL_MONO = None  # monotonic() timestamp of last successful upstream poll

# ===================== NEW: SQLite & Secrets ======================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_URL = f"sqlite:///{DATA_DIR.as_posix()}/app.db"

Base = declarative_base()
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Secret(Base):
    """
    Stores Argon2 hashes for:
      - pin.dashboard
      - pin.user
      - pin.admin   (fixed; API will not allow changing)
    """
    __tablename__ = "secrets"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

def _seed_secret_if_missing(db, key: str, plain: str):
    row = db.get(Secret, key)
    if not row:
        db.add(Secret(key=key, value=argon2.hash(plain)))
        db.commit()

def init_db_and_seed():
    Base.metadata.create_all(engine)
    db = SessionLocal()
    # Seed defaults (one-time)
    _seed_secret_if_missing(db, "pin.dashboard", "AT-MOD-01")
    _seed_secret_if_missing(db, "pin.user",      "AT-User-1")
    _seed_secret_if_missing(db, "pin.admin",     "AT1959")
# ================================================================

# ===================== NEW: JWT helpers ==========================
JWT_SECRET = os.environ.get("APP_JWT_SECRET", "CHANGE_ME")  # set env in prod
JWT_ALG = "HS256"
bearer = HTTPBearer(auto_error=True)

def issue_token(scope: str, hours: int = 8) -> str:
    payload = {"scope": scope, "exp": datetime.utcnow() + timedelta(hours=hours)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def require_scope(required: str):
    """
    Dependency: requires the JWT to have the given scope.
    'admin' can do everything.
    """
    def _inner(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
        try:
            payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALG])
        except JWTError:
            raise HTTPException(401, "Invalid or expired token")
        scope = payload.get("scope")
        if scope not in (required, "admin"):
            raise HTTPException(403, "Forbidden")
        return scope
    return _inner
# ================================================================


# ===================== SETTINGS (unchanged structure) ============
SETTINGS_PATH = "settings.json"

DEFAULT_SETTINGS = {
    "upstream": {  # master -> device on CH1 (serial params are fixed in code)
        "device_unit_id": 1,
        "poll_period_s": 1.0
    },
    "mirror_rtu": {  # slave on CH2 (hot-reload supported)
        "slave_id": 2,
        "baudrate": 9600,
        "parity": "N",
        "stopbits": 1,
        "bytesize": 8
    },
    "tcp": {  # Modbus TCP (hot-reload supported)
        "port": 1502
    },
    "local_units": {  # two views, both hot-reloadable
        "unit0_id": 1,  # 0-based @ 0..(count-1) -> dashboard
        "unit1_id": 2   # 1-based @ 1..count      -> Modbus Poll
    },
    "hr": {  # mirrored HR window (hot-reloadable)
        "start": 0,
        "count": 24
    },
    "branding": {
        "phone": "011-4639-8310",
        "email": "info@adaxtecna.com",
        "youtube": "https://www.youtube.com/@adaxtecna",
        "logo_url": "/static/adax_logo.png",
        "qr_url": ""
    },
    "device": { "model": "" }
}
# =====================================================

# ---------- Settings helpers ----------
SETTINGS_LOCK = asyncio.Lock()
SETTINGS: Dict[str, Any] = {}

# ==== Fixed upstream RTU (CH1) serial parameters ====
UP_CH1_PORT      = "/dev/ttySC0"
UP_CH1_BAUDRATE  = 9600
UP_CH1_PARITY    = "N"
UP_CH1_STOPBITS  = 1
UP_CH1_BYTESIZE  = 8
# ====================================================

# ==== Fixed Mirror RTU (CH2) serial port ====
MIRROR_CH2_PORT = "/dev/ttySC1"
# ==========================================

async def snapshot_regs() -> list[int]:
    """Return a safe snapshot of the 0-based HR window."""
    async with HR_LOCK:
        return _hr_block0().getValues(0, S()["hr"]["count"])

@asynccontextmanager
async def maybe_lock(lock):
    if lock is None:
        yield
        return
    if hasattr(lock, "__aenter__"):
        async with lock:  # type: ignore
            yield
        return
    if hasattr(lock, "acquire") and hasattr(lock, "release"):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lock.acquire)
        try:
            yield
        finally:
            lock.release()
        return
    yield

# ---------- Alarm & State model helpers ----------
def _bit(v: int, n: int) -> int:
    return 1 if (int(v) >> n) & 1 else 0

def build_measurements_from_hr(hr: list[int]) -> dict:
    return {
        "battery_voltage_v": round(hr[0] / 10.0, 1),
        "load_voltage_v": round(hr[1] / 10.0, 1),
        "battery_current_a": round(hr[2] / 10.0, 1),
        "load_current_a": round(hr[3] / 10.0, 1),
        "total_current_a": round(hr[4] / 10.0, 1),
        "ac_rn_v": hr[5],
        "ac_sn_v": hr[6],
        "ac_tn_v": hr[7],
        "ambient_temp_c": round(hr[8] / 10.0, 1),
        "ambient_temp_max_c": round(hr[9] / 10.0, 1)
    }

def build_alarms_from_bits(hr10: int, hr11: int) -> list[dict]:
    items = [
        {"key": "polo_tierra", "label": "POLO A TIERRA", "active": _bit(hr10, 0) == 1},
        {"key": "alta_v_bat", "label": "ALTA TENSIÓN BATERÍA", "active": _bit(hr10, 1) == 1},
        {"key": "baja_v_bat", "label": "BAJA TENSIÓN BATERÍA", "active": _bit(hr10, 2) == 1},
        {"key": "incom_consumo", "label": "INCOMUNICACIÓN CONSUMO", "active": _bit(hr11, 0) == 1},
        {"key": "red_ca_anormal", "label": "RED C.A. ANORMAL", "active": _bit(hr11, 3) == 1},
        {"key": "alta_v_consumo", "label": "ALTA TENSIÓN CONSUMO", "active": _bit(hr11, 4) == 1},
        {"key": "baja_v_consumo", "label": "BAJA TENSIÓN CONSUMO", "active": _bit(hr11, 5) == 1},
        {"key": "fusible_abierto", "label": "FUSIBLE ABIERTO", "active": _bit(hr11, 7) == 1},
        {"key": "alta_temp", "label": "ALTA TEMPERATURA", "active": _bit(hr11, 2) == 1},
    ]
    return items

def build_operating_state(hr10: int, hr11: int, hr12: int) -> list[dict]:
    return [
        {"key":"rectificador","label":"RECTIFICADOR",
         "value":"ENCENDIDO" if _bit(hr10,7) else "APAGADO",
         "color":"green" if _bit(hr10,7) else "red"},
        {"key":"bat_sentido","label":"BATERÍA EN",
         "value":"CARGA" if hr12==1 else "DESCARGA",
         "color":"green" if hr12==1 else "red"},
        {"key":"modo_carga","label":"MODO DE CARGA",
         "value":"MANUAL" if _bit(hr10,5) else "AUTOMÁTICO",
         "color":"orange" if _bit(hr10,5) else "green"},
        {"key":"nivel_carga","label":"NIVEL DE CARGA",
         "value":"FONDO" if _bit(hr10,4) else "FLOTE",
         "color":"black"},
        {"key":"timer_nicd","label":"TIMER NiCd",
         "value":"INICIADO" if _bit(hr11,6) else "DESACTIVADO",
         "color":"black"},
    ]

def _deep_merge(defs, cur):
    if isinstance(defs, dict):
        out = {}
        for k, v in defs.items():
            if k in cur:
                out[k] = _deep_merge(v, cur[k])
            else:
                out[k] = v
        for k, v in cur.items():
            if k not in out:
                out[k] = v
        return out
    else:
        return cur if cur is not None else defs

def load_settings_from_disk() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        return _deep_merge(DEFAULT_SETTINGS, data)
    except Exception:
        return DEFAULT_SETTINGS.copy()

async def save_settings_to_disk(settings: Dict[str, Any]):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)

SETTINGS = load_settings_from_disk()
def S(): return SETTINGS  # shorthand accessor

# ---------------- Datastore & context ----------------
HR_LOCK = asyncio.Lock()  # guards HR reads/write
store0: ModbusSlaveContext = None
store1: ModbusSlaveContext = None
tcp_context: ModbusServerContext = None      # used by Modbus TCP server
mirror_context: ModbusServerContext = None   # used by CH2 serial mirror

def make_store0(count: int) -> ModbusSlaveContext:
    return ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0]),
        co=ModbusSequentialDataBlock(0, [0]),
        hr=ModbusSequentialDataBlock(0, [0] * count),
        ir=ModbusSequentialDataBlock(0, [0])
    )

def make_store1(count: int) -> ModbusSlaveContext:
    # +1 so index 1..count is populated for 1-based clients
    return ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0]),
        co=ModbusSequentialDataBlock(0, [0]),
        hr=ModbusSequentialDataBlock(0, [0] * (count + 1)),
        ir=ModbusSequentialDataBlock(0, [0])
    )

def _hr_block0() -> ModbusSequentialDataBlock:
    return store0.store["h"]

def _hr_block1() -> ModbusSequentialDataBlock:
    return store1.store["h"]

def _copy_hr_values(src: List[int], dst: ModbusSequentialDataBlock, start_addr: int):
    dst.setValues(start_addr, src)

async def rebuild_datastores_and_context():
    global store0, store1, tcp_context, mirror_context

    hr_start = S()["hr"]["start"]
    hr_count = S()["hr"]["count"]
    u0 = S()["local_units"]["unit0_id"]
    u1 = S()["local_units"]["unit1_id"]
    mirror_id = (S().get("mirror_rtu", {}) or {}).get("slave_id", u1)

    # snapshot old data
    old0, old1 = [], []
    if store0 and store1:
        with contextlib.suppress(Exception):
            old0 = _hr_block0().getValues(0, hr_count)
        with contextlib.suppress(Exception):
            old1 = _hr_block1().getValues(1, hr_count)

    # new stores
    new0 = make_store0(hr_count)
    new1 = make_store1(hr_count)

    # copy overlap
    if old0:
        _copy_hr_values(old0[:hr_count], new0.store["h"], 0)
    if old1:
        _copy_hr_values(old1[:hr_count], new1.store["h"], 1)

    # swap + build contexts
    async with HR_LOCK:
        store0 = new0
        store1 = new1

        # TCP: unit0 (0-based) + unit1 (1-based)
        tcp_context = ModbusServerContext(slaves={u0: store0, u1: store1}, single=False)

        # CH2: ONLY mirror_id (1-based view)
        mirror_context = ModbusServerContext(slaves={mirror_id: store1}, single=False)

print(f"[MAP] TCP serves units: {S()['local_units']['unit0_id']}, {S()['local_units']['unit1_id']}")
print(f"[MAP] CH2 serves unit:  {(S().get('mirror_rtu',{}) or {}).get('slave_id')}")

async def _write_both_views(regs: List[int]):
    async with HR_LOCK:
        _hr_block0().setValues(0, regs)  # 0..count-1
        _hr_block1().setValues(1, regs)  # 1..count

# --------------- Scaling meta ---------------
ANNEX_A: Dict[int, Tuple[str, float]] = {
    0:  ("BATTERY_VOLTAGE_V", 10.0),
    1:  ("LOAD_VOLTAGE_V", 10.0),
    2:  ("BATTERY_CURRENT_A", 10.0),
    3:  ("LOAD_CURRENT_A", 10.0),
    4:  ("TOTAL_CURRENT_A", 10.0),
    5:  ("AC_VOLTAGE_RN_raw", 1.0),
    6:  ("AC_VOLTAGE_SN_raw", 1.0),
    7:  ("AC_VOLTAGE_TN_raw", 1.0),
    8:  ("AMBIENT_TEMP_C", 10.0),
    9:  ("AMBIENT_TEMP_MAX_C", 10.0),
    10: ("ALARM_BYTE_1_bits", 1.0),
    11: ("ALARM_BYTE_2_bits", 1.0),
    12: ("BATTERY_MODE_raw", 1.0),
    13: ("BATTERY_TEST_COUNT", 1.0),
    14: ("LAST_TEST_DURATION_H", 1.0),
    15: ("LAST_TEST_DURATION_MIN", 1.0),
    16: ("LAST_TEST_DAY", 1.0),
    17: ("LAST_TEST_MONTH", 1.0),
    18: ("LAST_TEST_YEAR", 1.0),
    19: ("LAST_TEST_FINAL_BATT_V", 10.0),
    20: ("FLOAT_CURRENT_SETPOINT_PbCa_A", 10.0),
    21: ("NUM_CELLS_NiCd", 1.0),
    22: ("TIMER_HOURS_NiCd", 1.0),
    23: ("SERIAL_NUMBER_raw", 1.0),
}

def _scaled(idx: int, raw: int) -> float:
    div = ANNEX_A.get(idx, ("", 1.0))[1]
    return raw / div if div and div != 1.0 else float(raw)

def _bits16(x: int) -> str:
    return format(x & 0xFFFF, "016b")

async def _safe_close(x):
    try:
        if x is None: return
        close_fn = getattr(x, "close", None)
        if close_fn:
            if inspect.iscoroutinefunction(close_fn): await close_fn()
            else: close_fn()
    except Exception:
        pass

# ================== Managers & Poller ==================
tcp_reload_event = asyncio.Event()
mirror_reload_event = asyncio.Event()

async def poll_upstream_and_update_cache():
    """Upstream master poller with live RTU reload."""
    client = None
    cur = {}

    async def connect():
        nonlocal client, cur
        ups = S()["upstream"]
        cur = {"device_unit_id": ups.get("device_unit_id", 1),
               "poll_period_s":  ups.get("poll_period_s", 1.0)}
        client = AsyncModbusSerialClient(
            port=UP_CH1_PORT,
            framer=FramerType.RTU,
            baudrate=UP_CH1_BAUDRATE,
            parity=UP_CH1_PARITY,
            stopbits=UP_CH1_STOPBITS,
            bytesize=UP_CH1_BYTESIZE,
            timeout=2
        )
        ok = await client.connect()
        if not ok:
            await _safe_close(client)
            client = None
            raise RuntimeError(f"Failed to open upstream port {UP_CH1_PORT}")

    def rtu_settings_changed() -> bool:
        return False

    await connect()

    try:
        while True:
            if rtu_settings_changed():
                await _safe_close(client)
                await connect()

            ups = S()["upstream"]
            hr = S()["hr"]
            rr = await client.read_holding_registers(
                address=hr["start"], count=hr["count"], slave=ups["device_unit_id"]
            )
            ts = time.strftime("%H:%M:%S")
            if rr and not rr.isError() and hasattr(rr, "registers"):
                regs = rr.registers[:hr["count"]]
                global LAST_GOOD_POLL_MONO
                LAST_GOOD_POLL_MONO = time.monotonic()

                await _write_both_views(regs)

                try:
                    bv = _scaled(0, regs[0]) if len(regs)>0 else float("nan")
                    lv = _scaled(1, regs[1]) if len(regs)>1 else float("nan")
                    bc = _scaled(2, regs[2]) if len(regs)>2 else float("nan")
                    lc = _scaled(3, regs[3]) if len(regs)>3 else float("nan")
                    tc = _scaled(4, regs[4]) if len(regs)>4 else float("nan")
                    temp = _scaled(8, regs[8]) if len(regs)>8 else float("nan")
                    a1 = regs[10] if len(regs)>10 else 0
                    a2 = regs[11] if len(regs)>11 else 0
                    mode = "charge" if (len(regs)>12 and regs[12]==1) else "discharge"
                    print(f"[{ts}] HR[0:{hr['count']}]: {regs}")
                    print(f"[{ts}] Vb={bv:.1f}V Vl={lv:.1f}V Ib={bc:.1f}A Il={lc:.1f}A It={tc:.1f}A T={temp:.1f}°C | "
                          f"A1={_bits16(a1)} A2={_bits16(a2)} | mode={mode}")
                except Exception:
                    print(f"[{ts}] HR[0:{hr['count']}]: {regs}")
            else:
                print(f"[{ts}] Read failed or exception from device")
            await asyncio.sleep(ups["poll_period_s"])
    finally:
        await _safe_close(client)

async def tcp_server_manager():
    """Hot-reloadable Modbus TCP server."""
    current_port = None
    current_ctx = None
    server_task: asyncio.Task | None = None

    async def _start(port: int, ctx: ModbusServerContext) -> asyncio.Task:
        async def run():
            await StartAsyncTcpServer(
                context=ctx,
                address=("0.0.0.0", port),
                ignore_missing_slaves=True,
            )
        return asyncio.create_task(run(), name=f"mbtcp:{port}")

    try:
        while True:
            desired_port = S()["tcp"]["port"]
            desired_ctx = tcp_context

            needs_restart = (
                desired_port != current_port or
                desired_ctx is not current_ctx or
                (server_task is not None and server_task.done())
            )

            if needs_restart:
                if server_task and not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task

                current_port = desired_port
                current_ctx = desired_ctx
                server_task = await _start(current_port, current_ctx)
                print(f"[TCP] listening on 0.0.0.0:{current_port}")

            await asyncio.sleep(0.5)
    finally:
        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task

async def mirror_rtu_server_manager():
    """Hot-reloadable Modbus RTU mirror on CH2."""
    current_serial = {}
    current_ctx = None
    server_task: asyncio.Task | None = None

    async def _start(cfg: dict, ctx: ModbusServerContext) -> asyncio.Task:
        async def run():
            await StartAsyncSerialServer(
                context=ctx,
                framer=FramerType.RTU,
                port=MIRROR_CH2_PORT,
                baudrate=cfg["baudrate"],
                parity=cfg["parity"],
                stopbits=cfg["stopbits"],
                bytesize=cfg["bytesize"],
                timeout=1
            )
        return asyncio.create_task(run(), name=f"mbserial:{MIRROR_CH2_PORT}")

    try:
        while True:
            cfg = S().get("mirror_rtu", {}) or {}
            watched = {
                "baudrate": cfg.get("baudrate"),
                "parity":   cfg.get("parity"),
                "stopbits": cfg.get("stopbits"),
                "bytesize": cfg.get("bytesize"),
            }
            desired_ctx = mirror_context

            needs_restart = (
                watched != current_serial or
                desired_ctx is not current_ctx or
                (server_task is None) or
                server_task.done()
            )

            if needs_restart:
                if server_task and not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task

                current_serial = watched
                current_ctx = desired_ctx
                server_task = await _start(cfg, current_ctx)
                print(f"[RTU mirror] {MIRROR_CH2_PORT} {cfg.get('baudrate')} {cfg.get('parity')} "
                      f"{cfg.get('stopbits')} {cfg.get('bytesize')} | slave_id={cfg.get('slave_id')}")

            await asyncio.sleep(0.5)
    finally:
        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task

# ================== Web API & Dashboard ==================

# ---------- NEW: Auth endpoints ----------
@app.post("/api/auth/login")
def auth_login(body: Dict[str, str] = Body(...)):
    """
    Body: {"who": "dashboard"|"user"|"admin", "password": "<plain>"}
    Returns: {"token": "<jwt>", "scope": "<who>"}
    """
    who = (body or {}).get("who", "")
    pwd = (body or {}).get("password", "")
    if who not in ("dashboard", "user", "admin"):
        raise HTTPException(400, "who must be one of: dashboard, user, admin")

    db = SessionLocal()
    row = db.get(Secret, f"pin.{who}")
    if not row or not argon2.verify(pwd, row.value):
        raise HTTPException(401, "Invalid credentials")

    return {"token": issue_token(scope=who), "scope": who}

@app.post("/api/auth/change")
def auth_change(body: Dict[str, str] = Body(...), _scope: str = Depends(require_scope("user"))):
    """
    Body: {"who": "dashboard"|"user", "old_password": "...", "new_password": "..."}
    Admin pin is FIXED and cannot be changed here.
    Any authenticated scope (user/dashboard/admin) can change dashboard/user (you can tighten if desired).
    """
    who = (body or {}).get("who", "")
    oldp = (body or {}).get("old_password", "")
    newp = (body or {}).get("new_password", "")

    if who not in ("dashboard", "user"):
        raise HTTPException(400, "Only dashboard or user pins are changeable")
    if not newp or len(newp) < 6 or len(newp) > 64:
        raise HTTPException(400, "new_password must be 6..64 chars")

    db = SessionLocal()
    key = f"pin.{who}"
    row = db.get(Secret, key)
    if not row or not argon2.verify(oldp, row.value):
        raise HTTPException(401, "Old password is incorrect")

    row.value = argon2.hash(newp)
    db.add(row); db.commit()
    return {"ok": True}
# ---------- end auth ----------

@app.get("/api/meas2")
async def api_meas2():
    try:
        hr = await snapshot_regs()
        def r(i, default=0): return hr[i] if i < len(hr) else default
        return {
            "battery_voltage_v": round(r(0) / 10.0, 1),
            "load_voltage_v":    round(r(1) / 10.0, 1),
            "battery_current_a": round(r(2) / 10.0, 1),
            "load_current_a":    round(r(3) / 10.0, 1),
            "total_current_a":   round(r(4) / 10.0, 1),
            "ac_rn_v":           int(r(5)),
            "ac_sn_v":           int(r(6)),
            "ac_tn_v":           int(r(7)),
            "ambient_temp_c":    round(r(8) / 10.0, 1),
            "ambient_temp_max_c":round(r(9) / 10.0, 1),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/alarms2")
async def api_alarms2():
    try:
        hr = await snapshot_regs()
        a1 = hr[10] if len(hr) > 10 else 0
        a2 = hr[11] if len(hr) > 11 else 0
        def bit(v, n): return 1 if ((int(v) >> n) & 1) else 0
        items = [
            {"key": "polo_tierra",     "label": "POLO A TIERRA",          "active": bit(a1,0)==1},
            {"key": "alta_v_bat",      "label": "ALTA TENSIÓN BATERÍA",   "active": bit(a1,1)==1},
            {"key": "baja_v_bat",      "label": "BAJA TENSIÓN BATERÍA",   "active": bit(a1,2)==1},
            {"key": "incom_consumo",   "label": "INCOMUNICACIÓN CONSUMO", "active": bit(a2,0)==1},
            {"key": "red_ca_anormal",  "label": "RED C.A. ANORMAL",       "active": bit(a2,3)==1},
            {"key": "alta_v_consumo",  "label": "ALTA TENSIÓN CONSUMO",   "active": bit(a2,4)==1},
            {"key": "baja_v_consumo",  "label": "BAJA TENSIÓN CONSUMO",   "active": bit(a2,5)==1},
            {"key": "fusible_abierto", "label": "FUSIBLE ABIERTO",        "active": bit(a2,7)==1},
            {"key": "alta_temp",       "label": "ALTA TEMPERATURA",       "active": bit(a2,2)==1},
        ]
        return {"items": items}
    except Exception as e:
        return {"items": [], "error": str(e)}

@app.get("/api/states2")
async def api_states2():
    try:
        hr = await snapshot_regs()
        a1 = hr[10] if len(hr) > 10 else 0
        a2 = hr[11] if len(hr) > 11 else 0
        hr12 = hr[12] if len(hr) > 12 else 0
        def bit(v, n): return 1 if ((int(v) >> n) & 1) else 0
        items = [
            {"key":"rectificador","label":"RECTIFICADOR",
             "value":"ENCENDIDO" if bit(a1,7) else "APAGADO",
             "color": "green" if bit(a1,7) else "red"},
            {"key":"bat_sentido","label":"BATERÍA EN",
             "value":"CARGA" if hr12==1 else "DESCARGA",
             "color":"green" if hr12==1 else "red"},
            {"key":"modo_carga","label":"MODO DE CARGA",
             "value":"MANUAL" if bit(a1,5) else "AUTOMÁTICO",
             "color":"orange" if bit(a1,5) else "green"},
            {"key":"nivel_carga","label":"NIVEL DE CARGA",
             "value":"FONDO" if bit(a1,4) else "FLOTE",
             "color":"black"},
            {"key":"timer_nicd","label":"TIMER NiCd",
             "value":"INICIADO" if bit(a2,6) else "DESACTIVADO",
             "color":"black"},
        ]
        return {"items": items}
    except Exception as e:
        return {"items": [], "error": str(e)}

@app.get("/api/alarms")
async def api_alarms():
    async with HR_LOCK:
        regs = _hr_block0().getValues(0, S()["hr"]["count"])

    scaled = {}
    for i, raw in enumerate(regs):
        label = ANNEX_A.get(i, (f"HR{i}", 1.0))[0]
        div   = ANNEX_A.get(i, ("", 1.0))[1]
        scaled[label] = (raw / div) if div and div != 1.0 else float(raw)

    ups = S()["upstream"]
    statuses = ALARM_ENGINE.evaluate(
        raw_regs=regs,
        scaled=scaled,
        last_good_poll_ts=LAST_GOOD_POLL_MONO,
        poll_period_s=float(ups.get("poll_period_s", 1.0)),
    )

    return JSONResponse({
        "alarms": [
            {"name": st.name, "on": st.on, "color": st.color, "msg": st.msg}
            for st in statuses
        ]
    })

@app.get("/api/hr")
async def api_hr():
    async with HR_LOCK:
        regs = _hr_block0().getValues(0, S()["hr"]["count"])
    out: Dict[str, Any] = {
        "raw": regs,
        "scaled": {},
        "meta": {"unit_id": S()["local_units"]["unit0_id"], "start": 0, "count": S()["hr"]["count"]}
    }
    for i, _ in enumerate(regs):
        label = ANNEX_A.get(i, (f"HR{i}", 1.0))[0]
        out["scaled"][label] = _scaled(i, regs[i])
    out["mode"] = "charge" if (len(regs)>12 and regs[12]==1) else "discharge"
    out["alarm_bits"] = {
        "ALARM_BYTE_1": _bits16(regs[10] if len(regs)>10 else 0),
        "ALARM_BYTE_2": _bits16(regs[11] if len(regs)>11 else 0),
    }
    return JSONResponse(out)

@app.get("/api/debug_store")
async def api_debug_store():
    async with HR_LOCK:
        b0 = _hr_block0().getValues(0, S()["hr"]["count"])
        b1 = _hr_block1().getValues(1, S()["hr"]["count"])
    return {"unit1_addr0..": b0, "unit2_addr1..": b1}

@app.get("/api/settings")
async def get_settings():
    async with SETTINGS_LOCK:
        s = json.loads(json.dumps(S()))
    up = s.get("upstream", {}) or {}
    s["upstream"] = {
        "device_unit_id": up.get("device_unit_id", 1),
        "poll_period_s":  up.get("poll_period_s", 1.0),
    }
    mr = s.get("mirror_rtu", {}) or {}
    s["mirror_rtu"] = {
        "slave_id": mr.get("slave_id", (s.get("local_units", {}) or {}).get("unit1_id", 2)),
        "baudrate": mr.get("baudrate", 9600),
        "parity":   mr.get("parity", "N"),
        "stopbits": mr.get("stopbits", 1),
        "bytesize": mr.get("bytesize", 8),
    }
    return JSONResponse(s)

@app.put("/api/settings")
async def put_settings(payload: Dict[str, Any] = Body(...), _=Depends(require_scope("admin"))):
    """
    Admin-only: update settings and hot-reload contexts if HR window / units / mirror slave change.
    """
    payload = payload or {}

    # ---- sanitize upstream
    if isinstance(payload.get("upstream"), dict):
        up_in = payload["upstream"]
        payload["upstream"] = {}
        if "device_unit_id" in up_in:
            payload["upstream"]["device_unit_id"] = int(up_in["device_unit_id"])
        if "poll_period_s" in up_in:
            payload["upstream"]["poll_period_s"] = float(up_in["poll_period_s"])

    # ---- sanitize mirror_rtu
    if isinstance(payload.get("mirror_rtu"), dict):
        mr_in = payload["mirror_rtu"]
        mr_out: Dict[str, Any] = {}
        if "slave_id" in mr_in:
            mr_out["slave_id"] = int(mr_in["slave_id"])
        for k in ("baudrate", "stopbits", "bytesize"):
            if k in mr_in: mr_out[k] = int(mr_in[k])
        if "parity" in mr_in: mr_out["parity"] = str(mr_in["parity"])
        payload["mirror_rtu"] = mr_out  # no 'port'

    async with SETTINGS_LOCK:
        current_on_disk = load_settings_from_disk()

        def deep_merge(dst, src):
            for k, v in (src or {}).items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v

        prev = json.loads(json.dumps(current_on_disk))
        deep_merge(current_on_disk, payload)

        # purge unsupported keys
        if isinstance(current_on_disk.get("upstream"), dict):
            for k in ("port", "baudrate", "parity", "stopbits", "bytesize"):
                current_on_disk["upstream"].pop(k, None)
        if isinstance(current_on_disk.get("mirror_rtu"), dict):
            current_on_disk["mirror_rtu"].pop("port", None)

        try:
            await save_settings_to_disk(current_on_disk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

        SETTINGS.clear()
        SETTINGS.update(current_on_disk)

    prev_mirror_slave = (prev.get("mirror_rtu", {}) or {}).get("slave_id")
    new_mirror_slave  = (S().get("mirror_rtu", {}) or {}).get("slave_id")
    need_rebuild = (
        (prev.get("hr", {}) or {}) != (S().get("hr", {}) or {}) or
        (prev.get("local_units", {}) or {}) != (S().get("local_units", {}) or {}) or
        prev_mirror_slave != new_mirror_slave
    )
    if need_rebuild:
        await rebuild_datastores_and_context()

    return JSONResponse({"ok": True})

# Serve external dashboard.html at root
@app.get("/")
async def root():
    return FileResponse("dashboard.html", media_type="text/html")

# ================ Server boot & tasks =================
async def start_web():
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    # NEW: initialize DB & seed PINs
    init_db_and_seed()

    # initial stores/context
    await rebuild_datastores_and_context()

    await asyncio.gather(
        poll_upstream_and_update_cache(),
        tcp_server_manager(),
        mirror_rtu_server_manager(),
        start_web(),
    )

if __name__ == "__main__":
    asyncio.run(main())
