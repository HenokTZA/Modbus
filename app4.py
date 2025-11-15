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
import socket, stat  # add with your imports

from alarms import AlarmsEngine
from pathlib import Path
from contextlib import asynccontextmanager

# ===================== NEW: Auth & DB imports =====================
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.hash import argon2
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, select

import errno, serial
import os

from typing import Optional
import logging
logging.getLogger("pymodbus").setLevel(logging.INFO)
logging.getLogger("pymodbus.framer.rtu").setLevel(logging.DEBUG)  # add near imports

import sys, signal, contextlib, subprocess
from pathlib import Path

try:
    # new-ish pymodbus exposes a server class with start/stop
    from pymodbus.server.async_io import ModbusTcpServer as _ModbusTcpServer
except Exception:
    _ModbusTcpServer = None

import inspect
try:
    from pymodbus.server.async_io import ModbusTcpServer as _ModbusTcpServer
except Exception:
    _ModbusTcpServer = None
from pymodbus.server import StartAsyncTcpServer  # you already have this

import contextlib
import asyncio

from typing import Optional
import ipaddress, subprocess, shlex, re

from fastapi import Body, HTTPException
import asyncio, ipaddress, subprocess, json
from typing import Any, Dict

PREV_MIRROR_ID: Optional[int] = None
PREV_EXPIRY: float = 0.0

SKIP_NEXT_WATCH_RELOAD = False


MIRROR_QUEUE: Optional[asyncio.Queue] = None
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




def _mask_to_prefix(mask: str) -> int:
    # allow "255.255.255.0" or "/24" or ""
    mask = (mask or "").strip()
    if not mask:
        return None  # caller decides default
    if mask.startswith("/"):
        return int(mask[1:])
    # dotted -> prefix
    try:
        net = ipaddress.IPv4Network(f"0.0.0.0/{mask}")
        return int(net.prefixlen)
    except Exception:
        raise HTTPException(400, f"Invalid netmask '{mask}'")

def _detect_primary_iface_and_ip() -> tuple[str, str]:
    """
    Returns (iface, ip) of the route used to reach the internet.
    """
    try:
        r = subprocess.run(["ip","route","get","8.8.8.8"], capture_output=True, text=True, check=True)
        line = r.stdout.strip().splitlines()[0]
        # e.g. "8.8.8.8 via 192.168.1.1 dev eth0 src 192.168.1.20 ..."
        dev = re.search(r"\bdev\s+(\S+)", line)
        src = re.search(r"\bsrc\s+(\S+)", line)
        iface = dev.group(1) if dev else ""
        ip = src.group(1) if src else ""
        return (iface, ip)
    except Exception:
        # fallback: pick settings iface, and try ip addr
        iface = (S().get("network",{}) or {}).get("iface","eth0")
        ip = ""
        try:
            r = subprocess.run(["ip","-4","addr","show",iface], capture_output=True, text=True)
            m = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", r.stdout)
            if m: ip = m.group(1)
        except Exception:
            pass
        return (iface, ip)

def _detect_default_gateway() -> str:
    try:
        r = subprocess.run(["ip","route","show","default"], capture_output=True, text=True)
        m = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", r.stdout)
        return m.group(1) if m else ""
    except Exception:
        return ""

def _detect_dns() -> list[str]:
    out = []
    try:
        with open("/etc/resolv.conf","r") as f:
            for line in f:
                if line.strip().startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        out.append(parts[1])
    except Exception:
        pass
    return out

def _validate_static(address: str, netmask: str, gateway: str, dns: list[str]) -> tuple[ipaddress.IPv4Interface, list[str]]:
    try:
        # allow address with /prefix OR with separate netmask
        if "/" in address:
            iface = ipaddress.IPv4Interface(address)
        else:
            pfx = _mask_to_prefix(netmask)
            if pfx is None:
                raise HTTPException(400, "Provide netmask (e.g. 255.255.255.0) or use address in CIDR form (e.g. 192.168.1.10/24)")
            iface = ipaddress.IPv4Interface(f"{address}/{pfx}")
    except Exception:
        raise HTTPException(400, f"Invalid IPv4 address '{address}'")

    try:
        gw = ipaddress.IPv4Address(gateway)
    except Exception:
        raise HTTPException(400, f"Invalid gateway '{gateway}'")

    if gw not in iface.network:
        raise HTTPException(400, "Gateway is not in the same subnet as the static address")

    dns_ok = []
    for d in dns or []:
        if not d: continue
        try:
            ipaddress.IPv4Address(d)
            dns_ok.append(d)
        except Exception:
            raise HTTPException(400, f"Invalid DNS '{d}'")

    return iface, dns_ok


def _force_close_tcp_port(port: int):
    """
    Close ONLY listening TCP sockets in this process bound to `port`
    (handles IPv4 and IPv6). Leaves established connections alone.
    """
    base = "/proc/self/fd"
    closed = 0
    for fdname in os.listdir(base):
        try:
            fd = int(fdname)
            st = os.fstat(fd)
        except Exception:
            continue
        if not stat.S_ISSOCK(st.st_mode):
            continue

        for family in (socket.AF_INET, socket.AF_INET6):
            s = None
            try:
                s = socket.fromfd(fd, family, socket.SOCK_STREAM)
                # Must be a bound/listening TCP socket
                laddr = s.getsockname()
                lport = (laddr[1] if isinstance(laddr, tuple) and len(laddr) >= 2 else None)
                if lport != port:
                    continue
                is_listening = s.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
                if not is_listening:
                    continue
                os.close(fd)
                closed += 1
                print(f"[TCP] forcibly closed listener fd={fd} on :{port}")
                break  # fd is gone; stop trying families
            except Exception:
                pass
            finally:
                if s is not None:
                    try: s.detach()
                    except Exception: pass
    if closed:
        print(f"[TCP] force-closed {closed} listening fd(s) on :{port}")






def _filter_user_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # mirror_rtu: allow slave_id + serial params
    mr_in = payload.get("mirror_rtu") or {}
    if isinstance(mr_in, dict):
        mr_out = {}
        for k in ("slave_id", "baudrate", "parity", "stopbits", "bytesize"):
            if k in mr_in:
                mr_out[k] = mr_in[k]
        if mr_out:
            out["mirror_rtu"] = mr_out

    # tcp: allow port only
    tcp_in = payload.get("tcp") or {}
    if isinstance(tcp_in, dict) and "port" in tcp_in:
        out["tcp"] = {"port": tcp_in["port"]}

    # local_units: allow unit1_id only
    lu_in = payload.get("local_units") or {}
    if isinstance(lu_in, dict) and "unit1_id" in lu_in:
        out["local_units"] = {"unit1_id": lu_in["unit1_id"]}

    # Allow switching CH2 mode + DNP3 link addresses
    ch2 = payload.get("ch2") or {}
    if isinstance(ch2, dict):
        out_ch2 = {}
        if "mode" in ch2:
            out_ch2["mode"] = ch2["mode"]
        dnp3 = ch2.get("dnp3") or {}
        if isinstance(dnp3, dict):
            dd = {}
            if "outstation_addr" in dnp3: dd["outstation_addr"] = dnp3["outstation_addr"]
            if "master_addr"     in dnp3: dd["master_addr"]     = dnp3["master_addr"]
            if dd: out_ch2["dnp3"] = dd
        if out_ch2: out["ch2"] = out_ch2


    # user cannot change: upstream, hr, branding, device, unit0_id
    return out



def _force_release_serial_fd(port_path: str = "/dev/ttySC1"):
    """
    Sweep our own process for any FDs still pointing at the serial device
    and close them. Safe because we only close descriptors whose symlink
    target is exactly the device node.
    """
    base = "/proc/self/fd"
    closed = 0
    try:
        for fdname in os.listdir(base):
            fpath = os.path.join(base, fdname)
            try:
                target = os.readlink(fpath)
            except OSError:
                continue
            if target == port_path:
                try:
                    os.close(int(fdname))
                    closed += 1
                except Exception:
                    pass
    except Exception as e:
        print(f"[RTU mirror] fd sweep error: {e}")
    if closed:
        print(f"[RTU mirror] forcibly closed {closed} lingering FD(s) for {port_path}")


async def settings_auto_reload():
    global SKIP_NEXT_WATCH_RELOAD
    path = Path(SETTINGS_PATH)
    last = path.stat().st_mtime if path.exists() else 0
    while True:
        try:
            cur = path.stat().st_mtime
            if cur != last:
                last = cur
                if SKIP_NEXT_WATCH_RELOAD:
                    SKIP_NEXT_WATCH_RELOAD = False
                else:
                    async with SETTINGS_LOCK:
                        SETTINGS.clear()
                        SETTINGS.update(load_settings_from_disk())
                    # Context only needs rebuild for hr/units/slave id
                    await rebuild_datastores_and_context()
                    print("[SETTINGS] reloaded from disk")
        except Exception as e:
            print("[SETTINGS] auto-reload error:", e)
        await asyncio.sleep(1.0)


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

def CH2_MODE() -> str:
    try:
        return (S().get("ch2", {}) or {}).get("mode", "modbus")
    except Exception:
        return "modbus"

def require_scopes(*allowed: str):
    def _inner(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
        try:
            payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALG])
        except JWTError:
            raise HTTPException(401, "Invalid or expired token")
        scope = payload.get("scope")
        if scope == "admin" or scope in allowed:
            return scope
        raise HTTPException(403, "Forbidden")
    return _inner


def _ctx_get_slave_map(ctx):
    """
    Return the internal slave map for a ModbusServerContext across pymodbus versions.
    Always returns a dict (or {} if unavailable).
    """
    if ctx is None:
        return {}

    # Try public attr 'slaves' (may be a dict *or* a callable in some versions)
    if hasattr(ctx, "slaves"):
        val = getattr(ctx, "slaves")
        if callable(val):
            try:
                val = val()  # some builds expose a callable returning the map
            except TypeError:
                val = None
        if isinstance(val, dict):
            return val

    # Try private attr
    val = getattr(ctx, "_slaves", None)
    if isinstance(val, dict):
        return val

    # Fallback: build from iteration if supported
    try:
        return {k: ctx[k] for k in list(ctx)}
    except Exception:
        return {}

def _ctx_set_slave_map(ctx, new_map: dict):
    """
    Set/replace the slave map on an existing ModbusServerContext across versions.
    Mutates in-place if possible; otherwise tries attribute rebinding.
    """
    if ctx is None:
        return
    # Update an existing dict in place if available
    for attr in ("slaves", "_slaves"):
        if hasattr(ctx, attr):
            cur = getattr(ctx, attr)
            if not callable(cur) and isinstance(cur, dict):
                cur.clear()
                cur.update(new_map)
                return
    # Try attribute rebinding (works if property has a setter or it's a plain attr)
    for attr in ("slaves", "_slaves"):
        try:
            setattr(ctx, attr, dict(new_map))
            return
        except Exception:
            pass
    # Last resort: best-effort per-key assignment (may not remove old keys)
    try:
        for k, v in new_map.items():
            ctx.__setitem__(k, v)
    except Exception:
        pass



def require_any_scope(allowed: list[str]):
    def _inner(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> str:
        try:
            payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALG])
        except JWTError:
            raise HTTPException(401, "Invalid or expired token")
        scope = payload.get("scope")
        if scope not in allowed:
            raise HTTPException(403, "Forbidden")
        return scope
    return _inner


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
    "ch2": {                    # NEW: what to run on RS485 CH2
        "mode": "modbus",       # "modbus" or "dnp3"
        "dnp3": {
            "outstation_addr": 100,   # DNP3 link-layer outstation address
            "master_addr": 1          # master address to accept
        }
    },
    "network": {
        "mode": "dhcp",          # "dhcp" (default) or "static"
        "iface": "eth0",         # change if your primary NIC is different
        "static": {
            "address": "",       # e.g. "192.168.1.100"
            "netmask": "",       # e.g. "255.255.255.0" (or leave empty if you’ll send /prefix)
            "gateway": "",       # e.g. "192.168.1.1"
            "dns": ["8.8.8.8","1.1.1.1"]
        }
    },
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

def _force_close_tcp_port(port: int):
    """
    Close ONLY listening TCP sockets in this process bound to `port`
    (handles IPv4 and IPv6). Leaves established connections alone.
    """
    base = "/proc/self/fd"
    closed = 0
    for fdname in os.listdir(base):
        try:
            fd = int(fdname)
            st = os.fstat(fd)
        except Exception:
            continue
        if not stat.S_ISSOCK(st.st_mode):
            continue

        for family in (socket.AF_INET, socket.AF_INET6):
            s = None
            try:
                s = socket.fromfd(fd, family, socket.SOCK_STREAM)
                # Must be a bound/listening TCP socket
                laddr = s.getsockname()
                lport = (laddr[1] if isinstance(laddr, tuple) and len(laddr) >= 2 else None)
                if lport != port:
                    continue
                is_listening = s.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
                if not is_listening:
                    continue
                os.close(fd)
                closed += 1
                print(f"[TCP] forcibly closed listener fd={fd} on :{port}")
                break  # fd is gone; stop trying families
            except Exception:
                pass
            finally:
                if s is not None:
                    try: s.detach()
                    except Exception: pass
    if closed:
        print(f"[TCP] force-closed {closed} listening fd(s) on :{port}")


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

    hr_count = S()["hr"]["count"]
    u0 = S()["local_units"]["unit0_id"]
    u1 = S()["local_units"]["unit1_id"]
    mirror_id = (S().get("mirror_rtu", {}) or {}).get("slave_id", u1)

    # snapshot current values so we don't lose HR data across rebuilds
    old0, old1 = [], []
    if store0 and store1:
        with contextlib.suppress(Exception):
            old0 = _hr_block0().getValues(0, hr_count)
        with contextlib.suppress(Exception):
            old1 = _hr_block1().getValues(1, hr_count)

    # new stores sized to the current HR window
    new0 = make_store0(hr_count)
    new1 = make_store1(hr_count)

    # copy overlap
    if old0:
        _copy_hr_values(old0[:hr_count], new0.store["h"], 0)
    if old1:
        _copy_hr_values(old1[:hr_count], new1.store["h"], 1)

    # desired slave maps (authoritative truth for this rebuild)
    tcp_slaves = {u0: new0, u1: new1}
    if mirror_id not in tcp_slaves:
        tcp_slaves[mirror_id] = new1

    # Grace period: keep previous mirror ID valid on TCP as well
    #if PREV_MIRROR_ID and PREV_MIRROR_ID != mirror_id and time.monotonic() < PREV_EXPIRY:
        #tcp_slaves.setdefault(PREV_MIRROR_ID, new1)

    mirror_map = {mirror_id: new1}
    #if PREV_MIRROR_ID and PREV_MIRROR_ID != mirror_id and time.monotonic() < PREV_EXPIRY:
        #mirror_map[PREV_MIRROR_ID] = new1
    #_ctx_set_slave_map(mirror_context, mirror_map)

    async with HR_LOCK:
        # swap global stores
        store0 = new0
        store1 = new1

        # ---- TCP context: create once, then mutate via helper (do NOT replace object)
        if tcp_context is None:
            tcp_context = ModbusServerContext(slaves=dict(tcp_slaves), single=False)
        else:
            _ctx_set_slave_map(tcp_context, tcp_slaves)

        # ---- CH2 mirror context: create once, then mutate via helper (do NOT replace object)
        if mirror_context is None:
            mirror_context = ModbusServerContext(slaves=dict(mirror_map), single=False)
        else:
            _ctx_set_slave_map(mirror_context, mirror_map)

    # logging: use the maps we just built (don’t introspect ctx; supports all versions)
    try:
        served_tcp = ", ".join(str(x) for x in sorted(tcp_slaves.keys()))
    except Exception:
        served_tcp = f"{list(tcp_slaves.keys())}"
    print(f"[MAP] TCP serves units: {served_tcp}")
    print(f"[MAP] CH2 serves unit:  {mirror_id}")






print(f"[MAP] TCP serves units: {S()['local_units']['unit0_id']}, {S()['local_units']['unit1_id']}")
print(f"[MAP] CH2 serves unit:  {(S().get('mirror_rtu',{}) or {}).get('slave_id')}")

async def _write_both_views(regs: List[int]):
    async with HR_LOCK:
        _hr_block0().setValues(0, regs)  # 0..count-1
        _hr_block1().setValues(1, regs)  # 1..count
    # push to sidecar (coalesce)
    try:
        if MIRROR_QUEUE is not None:
            # keep only the newest snapshot
            while not MIRROR_QUEUE.empty():
                MIRROR_QUEUE.get_nowait()
            MIRROR_QUEUE.put_nowait(list(regs))
    except Exception:
        pass


import sys, subprocess, contextlib

"""
async def mirror_sidecar_supervisor():
    child_proc: Optional[asyncio.subprocess.Process] = None
    child_stdin: Optional[asyncio.StreamWriter] = None
    current_cfg: Optional[dict] = None

    child_path = str(Path(__file__).parent / "mirror_sidecar.py")

    async def _stop_child():
        nonlocal child_proc, child_stdin
        if child_proc:
            with contextlib.suppress(Exception):
                child_proc.terminate()
            try:
                await asyncio.wait_for(child_proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    child_proc.kill()
            child_proc = None
        if child_stdin:
            with contextlib.suppress(Exception):
                child_stdin.close()
            child_stdin = None
        await asyncio.sleep(0.1)

    async def _start_child(cfg: dict):
        nonlocal child_proc, child_stdin
        args = [
            sys.executable, "-u", child_path,
            "--port", MIRROR_CH2_PORT,
            "--baudrate", str(cfg["baudrate"]),
            "--parity",   str(cfg["parity"]).upper()[:1],
            "--stopbits", str(cfg["stopbits"]),
            "--bytesize", str(cfg["bytesize"]),
            "--slave-id", str(cfg["slave_id"]),
            "--count",    str(S()["hr"]["count"]),
        ]
        child_proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        # In 3.9, this is already a StreamWriter:
        child_stdin = child_proc.stdin
        print(f"[RTU sidecar] spawned pid={child_proc.pid} cfg={cfg}")

    async def _send_snapshot(regs: List[int]):
        nonlocal child_stdin
        if not child_stdin:
            return
        msg = {"op": "snap", "values": [int(x) & 0xFFFF for x in regs]}
        data = (json.dumps(msg) + "\n").encode()
        child_stdin.write(data)
        with contextlib.suppress(Exception):
            await child_stdin.drain()

    def _desired_cfg() -> dict:
        mr = (S().get("mirror_rtu", {}) or {})
        return {
            "baudrate": int(mr.get("baudrate", 9600)),
            "parity":   str(mr.get("parity", "N")),
            "stopbits": int(mr.get("stopbits", 1)),
            "bytesize": int(mr.get("bytesize", 8)),
            "slave_id": int(mr.get("slave_id", (S().get("local_units", {}) or {}).get("unit1_id", 2))),
        }

    async def _initial_snapshot():
        try:
            regs = await snapshot_regs()
            await _send_snapshot(regs)
        except Exception:
            pass

    # ensure first bring-up happens quickly
    evt = mirror_reload_event or getattr(app.state, "mirror_reload_event", None)
    if evt:
        evt.set()

    while True:
        # Wait for a poke or light tick
        evt = mirror_reload_event or getattr(app.state, "mirror_reload_event", None)
        if evt:
            try:
                await asyncio.wait_for(evt.wait(), timeout=0.5)
                if evt.is_set():
                    evt.clear()
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(0.5)

        desired = _desired_cfg()
        need_restart = (
            current_cfg != desired or
            child_proc is None or
            (child_proc.returncode is not None)
        )

        if need_restart:
            await _stop_child()
            await _start_child(desired)
            current_cfg = desired
            await asyncio.sleep(0.2)
            await _initial_snapshot()

        # coalesce & send latest snapshot if queued
        if MIRROR_QUEUE is not None and not MIRROR_QUEUE.empty():
            last = None
            while not MIRROR_QUEUE.empty():
                last = await MIRROR_QUEUE.get()
            if last is not None:
                await _send_snapshot(last)

        await asyncio.sleep(0.05)
"""

async def ch2_supervisor():
    """
    Owns the RS485 CH2 process. Depending on S()['ch2']['mode'], it will spawn:
      - mirror_sidecar.py  (Modbus RTU mirror)
      - dnp3_sidecar.py    (DNP3 outstation)
    It streams HR snapshots to the child via JSON lines: {"op":"snap","values":[...]}
    """
    child_proc: Optional[asyncio.subprocess.Process] = None
    child_stdin: Optional[asyncio.StreamWriter] = None
    current_cfg: Optional[dict] = None  # includes mode + serial + ids
    child_path_modbus = str(Path(__file__).parent / "mirror_sidecar.py")
    child_path_dnp3   = str(Path(__file__).parent / "dnp3_sidecar.py")

    async def _stop_child():
        nonlocal child_proc, child_stdin
        if child_proc:
            with contextlib.suppress(Exception):
                child_proc.terminate()
            try:
                await asyncio.wait_for(child_proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    child_proc.kill()
            child_proc = None
        if child_stdin:
            with contextlib.suppress(Exception):
                child_stdin.close()
            child_stdin = None
        await asyncio.sleep(0.1)

    async def _start_child(cfg: dict):
        nonlocal child_proc, child_stdin
        mode = cfg["mode"]
        if mode == "modbus":
            args = [
                sys.executable, "-u", child_path_modbus,
                "--port", MIRROR_CH2_PORT,
                "--baudrate", str(cfg["baudrate"]),
                "--parity",   str(cfg["parity"]).upper()[:1],
                "--stopbits", str(cfg["stopbits"]),
                "--bytesize", str(cfg["bytesize"]),
                "--slave-id", str(cfg["slave_id"]),
                "--count",    str(S()["hr"]["count"]),
            ]
        else:  # dnp3
            args = [
                sys.executable, "-u", child_path_dnp3,
                "--port", MIRROR_CH2_PORT,
                "--baudrate", str(cfg["baudrate"]),
                "--parity",   str(cfg["parity"]).upper()[:1],
                "--stopbits", str(cfg["stopbits"]),
                "--bytesize", str(cfg["bytesize"]),
                "--outstation", str(cfg["outstation_addr"]),
                "--master",     str(cfg["master_addr"]),
                "--count",      str(S()["hr"]["count"]),
            ]
        child_proc = await asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        child_stdin = child_proc.stdin
        print(f"[CH2 {mode}] spawned pid={child_proc.pid} cfg={cfg}")

    async def _send_snapshot(regs: List[int]):
        nonlocal child_stdin
        if not child_stdin:
            return
        msg = {"op": "snap", "values": [int(x) & 0xFFFF for x in regs]}
        data = (json.dumps(msg) + "\n").encode()
        child_stdin.write(data)
        with contextlib.suppress(Exception):
            await child_stdin.drain()

    def _desired_cfg() -> dict:
        mr = (S().get("mirror_rtu", {}) or {})
        ch2 = (S().get("ch2", {}) or {})
        dnp = (ch2.get("dnp3", {}) or {})
        return {
            "mode": str(ch2.get("mode", "modbus")).lower(),
            "baudrate": int(mr.get("baudrate", 9600)),
            "parity":   str(mr.get("parity", "N")),
            "stopbits": int(mr.get("stopbits", 1)),
            "bytesize": int(mr.get("bytesize", 8)),
            "slave_id": int(mr.get("slave_id", (S().get("local_units", {}) or {}).get("unit1_id", 2))),
            "outstation_addr": int(dnp.get("outstation_addr", 100)),
            "master_addr":     int(dnp.get("master_addr", 1)),
        }

    async def _initial_snapshot():
        try:
            regs = await snapshot_regs()
            await _send_snapshot(regs)
        except Exception:
            pass

    # kick first start
    evt = mirror_reload_event or getattr(app.state, "mirror_reload_event", None)
    if evt: evt.set()

    while True:
        # Wait for a poke or tick
        evt = mirror_reload_event or getattr(app.state, "mirror_reload_event", None)
        if evt:
            try:
                await asyncio.wait_for(evt.wait(), timeout=0.5)
                if evt.is_set(): evt.clear()
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(0.5)

        desired = _desired_cfg()
        need_restart = (
            (current_cfg != desired) or
            (child_proc is None) or
            (child_proc.returncode is not None)
        )
        if need_restart:
            await _stop_child()
            # (optional) prime serial like before if you want:
            # await prime_serial_port(desired)
            await _start_child(desired)
            current_cfg = desired
            await asyncio.sleep(0.2)
            await _initial_snapshot()

        # coalesce & send latest snapshot
        if MIRROR_QUEUE is not None and not MIRROR_QUEUE.empty():
            last = None
            while not MIRROR_QUEUE.empty():
                last = await MIRROR_QUEUE.get()
            if last is not None:
                await _send_snapshot(last)

        await asyncio.sleep(0.05)



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
tcp_reload_event: Optional[asyncio.Event] = None
mirror_reload_event: Optional[asyncio.Event] = None

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
    """
    Robust Modbus TCP manager:
      - Uses defer_start when available;
      - On legacy pymodbus, captures ONLY the asyncio.Server created for the
        exact Modbus port, then closes it on port change.
    """
    current_port: Optional[int] = None
    server_obj = None
    legacy_server: Optional[asyncio.base_events.Server] = None
    server_task: Optional[asyncio.Task] = None

    async def _stop_old_listener(old_port: Optional[int]):
        nonlocal server_obj, legacy_server, server_task
        # Modern path
        if server_obj is not None:
            try:
                if hasattr(server_obj, "shutdown") and callable(server_obj.shutdown):
                    await server_obj.shutdown()
            except Exception:
                pass
            try:
                srv = getattr(server_obj, "server", None)
                if srv is not None:
                    srv.close()
                    await srv.wait_closed()
            except Exception:
                pass
            server_obj = None

        # Legacy path
        if legacy_server is not None:
            try:
                legacy_server.close()
                await legacy_server.wait_closed()
            except Exception:
                pass
            legacy_server = None

        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task
        server_task = None

        if old_port is not None:
            _force_close_tcp_port(int(old_port))
        await asyncio.sleep(0.2)

    async def _start_new_listener(new_port: int):
        nonlocal server_obj, legacy_server, server_task
        server_obj = None
        legacy_server = None
        server_task = None

        # Try modern path first
        try:
            srv = await StartAsyncTcpServer(
                context=tcp_context,
                address=("0.0.0.0", new_port),
                ignore_missing_slaves=False,
                defer_start=True,  # modern API
            )
            await srv.start()
            server_obj = srv
            print(f"[TCP] listening on 0.0.0.0:{new_port}")
            return
        except TypeError:
            pass  # legacy path

        # ---- Legacy path (scoped, port-matching capture) ----
        orig_start_server = asyncio.start_server
        captured_evt = asyncio.Event()

        async def spy_start_server(client_connected_cb, host=None, port=None, *args, **kwargs):
            # Only capture if THIS call is for the Modbus port we are starting
            if port == new_port:
                asyncio.start_server = orig_start_server  # restore immediately
                srv = await orig_start_server(client_connected_cb, host, port, *args, **kwargs)
                nonlocal legacy_server
                legacy_server = srv
                captured_evt.set()
                return srv
            # Not our server; just forward (do NOT capture)
            return await orig_start_server(client_connected_cb, host, port, *args, **kwargs)

        # Install the spy and ensure we restore it soon no matter what
        asyncio.start_server = spy_start_server

        async def run_legacy():
            print("[TCP] Legacy pymodbus path in use; consider upgrading.")
            try:
                await StartAsyncTcpServer(
                    context=tcp_context,
                    address=("0.0.0.0", new_port),
                    ignore_missing_slaves=False,
                )
            finally:
                # Safety: ensure spy is removed even if something throws
                if asyncio.start_server is spy_start_server:
                    asyncio.start_server = orig_start_server

        server_task = asyncio.create_task(run_legacy(), name=f"mbtcp:{new_port}")

        # Wait briefly for the capture; then make sure we restore the spy
        try:
            await asyncio.wait_for(captured_evt.wait(), timeout=1.5)
        except asyncio.TimeoutError:
            pass  # we may still be fine; just restore the spy now
        finally:
            if asyncio.start_server is spy_start_server:
                asyncio.start_server = orig_start_server

        print(f"[TCP] listening on 0.0.0.0:{new_port}")

    try:
        while True:
            desired_port = int(S()["tcp"]["port"])

            if current_port is None:
                await _start_new_listener(desired_port)
                current_port = desired_port
            elif desired_port != current_port:
                old_port = current_port
                await _stop_old_listener(old_port)
                await _start_new_listener(desired_port)
                current_port = desired_port

            await asyncio.sleep(0.4)
    finally:
        await _stop_old_listener(current_port)




async def prime_serial_port(cfg: dict):
    """Open with new params, flush, toggle lines, close — to ensure a clean state."""
    def _do():
        s = serial.Serial(
            port=MIRROR_CH2_PORT,
            baudrate=int(cfg["baudrate"]),
            parity=str(cfg["parity"]),
            stopbits=int(cfg["stopbits"]),
            bytesize=int(cfg["bytesize"]),
            timeout=0.05,
            exclusive=True,
        )
        try:
            # clear any stale bytes and poke line state
            s.reset_input_buffer()
            s.reset_output_buffer()
            with contextlib.suppress(Exception):
                s.dtr = False; s.rts = False
            time.sleep(0.05)
            with contextlib.suppress(Exception):
                s.dtr = True; s.rts = True
        finally:
            s.close()
    try:
        await asyncio.to_thread(_do)
    except Exception as e:
        print("[RTU mirror] prime open failed:", e)




# ---- helper: wait until /dev/ttySC1 can be opened exclusively ----
async def wait_port_free(port: str, timeout: float = 5.0, probe_baud: int = 9600) -> bool:
    """
    Returns True as soon as the port can be opened with exclusive=True (and closes it),
    or False after timeout. Non-blocking: uses asyncio.to_thread for the open.
    """
    last_err = None
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            def _probe():
                s = serial.Serial(port=port, baudrate=probe_baud, timeout=0.05, exclusive=True)
                try:
                    # on Linux this will fail with EBUSY/EAGAIN if someone still holds it
                    return True
                finally:
                    try: s.close()
                    except: pass
            ok = await asyncio.to_thread(_probe)
            if ok:
                return True
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.1)  # brief backoff

    print(f"[RTU mirror] port '{port}' still busy after {timeout:.1f}s: {last_err}")
    return False


# ================== Web API & Dashboard ==================

# ---------- NEW: Auth endpoints ----------

@app.get("/api/runtime/serial_status")
def serial_status():
    mr = (S().get("mirror_rtu", {}) or {})
    return {
        "port": MIRROR_CH2_PORT,
        "mode": CH2_MODE(),   # NEW
        "configured": {
            "baudrate": int(mr.get("baudrate", 9600)),
            "parity":   str(mr.get("parity", "N")).upper()[:1],
            "stopbits": int(mr.get("stopbits", 1)),
            "bytesize": int(mr.get("bytesize", 8)),
        }
    }


@app.get("/api/network")
def api_network_get(_=Depends(require_any_scope(["admin","user","dashboard"]))):
    s = S().get("network", {}) or {}
    iface_current, ip_current = _detect_primary_iface_and_ip()
    gw_current = _detect_default_gateway()
    dns_current = _detect_dns()

    saved = {
        "mode": s.get("mode","dhcp"),
        "iface": s.get("iface", iface_current or "eth0"),
        "static": {
            "address": ((s.get("static") or {}).get("address") or ""),
            "netmask": ((s.get("static") or {}).get("netmask") or ""),
            "gateway": ((s.get("static") or {}).get("gateway") or ""),
            "dns":     ((s.get("static") or {}).get("dns") or ["8.8.8.8","1.1.1.1"]),
        }
    }
    return {
        "current": {"iface": iface_current, "ip": ip_current, "gateway": gw_current, "dns": dns_current},
        "saved": saved,
        "note": "DHCP is default. Switching to static may disconnect your browser if IP/network changes."
    }

@app.put("/api/network")
async def api_network_put(body: Dict[str, Any] = Body(...)):
    body = body or {}

    mode  = str(body.get("mode", "dhcp")).lower().strip()
    iface = (
        body.get("iface")
        or ((S().get("network") or {}).get("iface"))
        or "eth0"
    )

    if mode not in ("dhcp", "static"):
        raise HTTPException(status_code=400, detail="mode must be 'dhcp' or 'static'")

    # ---------- STATIC ----------
    if mode == "static":
        st = body.get("static") or {}
        address = (st.get("address") or "").strip()
        netmask = (st.get("netmask") or "").strip()     # allow dotted or /prefix
        gateway = (st.get("gateway") or "").strip()
        dns     = st.get("dns") or []

        # your validator should return (IPv4Interface, list[str] or None)
        iface_if, dns_ok = _validate_static(address, netmask, gateway, dns)
        addr_cidr = str(iface_if.with_prefixlen)  # "a.b.c.d/pfx"

        # persist to settings.json (under the async lock)
        async with SETTINGS_LOCK:
            current = load_settings_from_disk()
            current.setdefault("network", {})
            current["network"]["mode"]  = "static"
            current["network"]["iface"] = iface
            current["network"]["static"] = {
                "address": str(iface_if.ip),
                "netmask": str(iface_if.network.netmask),
                "gateway": gateway,
                "dns": dns_ok or ["8.8.8.8", "1.1.1.1"],
            }
            await save_settings_to_disk(current)
            SETTINGS.clear()
            SETTINGS.update(current)

        # apply via helper script
        dns_csv = ",".join(dns_ok) if dns_ok else ""
        try:
            subprocess.run(
                ["sudo", "/usr/local/bin/netcfg-apply", "static", iface, addr_cidr, gateway, dns_csv],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to apply static IP: {e}")

        return {
            "ok": True,
            "applied": {
                "mode": "static",
                "iface": iface,
                "address": str(iface_if.ip),
                "netmask": str(iface_if.network.netmask),
                "gateway": gateway,
                "dns": dns_ok,
            },
            "note": "Applied static network. You may need to reconnect to the new IP.",
        }

    # ---------- DHCP ----------
    async with SETTINGS_LOCK:
        current = load_settings_from_disk()
        current.setdefault("network", {})
        current["network"]["mode"]  = "dhcp"
        current["network"]["iface"] = iface
        await save_settings_to_disk(current)
        SETTINGS.clear()
        SETTINGS.update(current)

    try:
        subprocess.run(["sudo", "/usr/local/bin/netcfg-apply", "dhcp", iface], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch to DHCP: {e}")

    return {
        "ok": True,
        "applied": {"mode": "dhcp", "iface": iface},
        "note": "Switched to DHCP. The IP may change; you may lose connection.",
    }

@app.get("/api/runtime/serial_status")
def serial_status():
    # very lightweight: what the app thinks it’s running with
    mr = (S().get("mirror_rtu", {}) or {})
    return {
        "port": MIRROR_CH2_PORT,
        "configured": {
            "baudrate": int(mr.get("baudrate", 9600)),
            "parity":   str(mr.get("parity", "N")).upper()[:1],
            "stopbits": int(mr.get("stopbits", 1)),
            "bytesize": int(mr.get("bytesize", 8)),
        }
    }



@app.get("/api/runtime/mirror_units")
def runtime_mirror_units():
    try:
        return {"mirror_units": sorted(_ctx_get_slave_map(mirror_context).keys())}
    except Exception as e:
        return {"mirror_units": [], "error": str(e)}



@app.post("/api/auth/login")
def auth_login(body: Dict[str, str] = Body(...)):
    # accept both styles
    who = (body or {}).get("who") or (body or {}).get("scope")
    pwd = (body or {}).get("password") or (body or {}).get("pin")
    if who == "gate":
        who = "dashboard"
    if who not in ("dashboard", "user", "admin"):
        raise HTTPException(400, "who must be one of: dashboard, user, admin")

    db = SessionLocal()
    row = db.get(Secret, f"pin.{who}")
    if not row or not argon2.verify(pwd or "", row.value):
        raise HTTPException(401, "Invalid credentials")

    return {"token": issue_token(scope=who), "scope": who}





@app.post("/api/auth/change")
def auth_change(
    body: Dict[str, str] = Body(...),
    _scope: str = Depends(require_any_scope(["dashboard", "user", "admin"]))
):
    """
    Body (accepts either pair of names):
      { who: "dashboard"|"user", old_password|old_pin: "...", new_password|new_pin: "..." }
    Admin pin is fixed and not changeable here.
    """
    who = (body or {}).get("who") or (body or {}).get("role")  # accept legacy 'role'
    oldp = (body or {}).get("old_password") or (body or {}).get("old_pin")
    newp = (body or {}).get("new_password") or (body or {}).get("new_pin")

    if who == "gate":  # normalize
        who = "dashboard"

    if who not in ("dashboard", "user"):
        raise HTTPException(400, "Only 'dashboard' or 'user' pins are changeable")
    if not newp or len(newp) < 6 or len(newp) > 64:
        raise HTTPException(400, "new_password must be 6..64 chars")

    db = SessionLocal()
    key = f"pin.{who}"
    row = db.get(Secret, key)
    if not row or not argon2.verify(oldp or "", row.value):
        raise HTTPException(401, "Old password is incorrect")

    row.value = argon2.hash(newp)
    row.updated_at = datetime.utcnow()
    db.add(row); db.commit()
    return {"ok": True}



@app.post("/api/auth/reset")
def auth_reset(_=Depends(require_any_scope(["dashboard","user","admin"]))):
    db = SessionLocal()
    defaults = {
        "pin.dashboard": "AT-MOD-01",
        "pin.user":      "AT-User-1",
        "pin.admin":     "AT1959",   # fixed
    }
    for k, v in defaults.items():
        row = db.get(Secret, k)
        if row:
            row.value = argon2.hash(v); row.updated_at = datetime.utcnow(); db.add(row)
        else:
            db.add(Secret(key=k, value=argon2.hash(v)))
    db.commit()
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
    ch2 = s.get("ch2", {}) or {}
    dnp = ch2.get("dnp3", {}) or {}
    s["ch2"] = {
        "mode": ch2.get("mode", "modbus"),
        "dnp3": {
            "outstation_addr": int(dnp.get("outstation_addr", 100)),
            "master_addr":     int(dnp.get("master_addr", 1)),
        }
    }
    s["mirror_rtu"] = {
        "slave_id": mr.get("slave_id", (s.get("local_units", {}) or {}).get("unit1_id", 2)),
        "baudrate": mr.get("baudrate", 9600),
        "parity":   mr.get("parity", "N"),
        "stopbits": mr.get("stopbits", 1),
        "bytesize": mr.get("bytesize", 8),
    }
    return JSONResponse(s)



@app.put("/api/settings")
async def put_settings(
    payload: Dict[str, Any] = Body(...),
    scope: str = Depends(require_any_scope(["admin", "user"]))
):

    payload = payload or {}

    # put this right after you read the request JSON and current settings
    mirror_serial_changed: bool = False
    mode_changed: bool = False
    dnp_changed: bool = False  # if you want to watch DNP3 addr changes


    if scope == "user":
        payload = _filter_user_payload(payload)
        # If nothing user-changeable was sent, just acknowledge.
        if not payload:
            return JSONResponse({"ok": True})

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
            if k in mr_in:
                mr_out[k] = int(mr_in[k])
        if "parity" in mr_in:
            mr_out["parity"] = str(mr_in["parity"])
        payload["mirror_rtu"] = mr_out  # no 'port'

    # ---- sanitize tcp
    if isinstance(payload.get("tcp"), dict):
        tcp_in = payload["tcp"]
        tcp_out: Dict[str, Any] = {}
        if "port" in tcp_in:
            try:
                tcp_out["port"] = int(tcp_in["port"])
            except Exception:
                raise HTTPException(status_code=400, detail="tcp.port must be an integer")
        payload["tcp"] = tcp_out

    # ---- sanitize ch2 (mode + dnp3 addrs)
    if isinstance(payload.get("ch2"), dict):
        ch2_in  = payload["ch2"]
        ch2_out: Dict[str, Any] = {}
        if "mode" in ch2_in:
            mode = str(ch2_in["mode"]).lower()
            if mode not in ("modbus", "dnp3"):
                raise HTTPException(400, "ch2.mode must be 'modbus' or 'dnp3'")
            ch2_out["mode"] = mode
        if isinstance(ch2_in.get("dnp3"), dict):
            di = ch2_in["dnp3"]; do: Dict[str, Any] = {}
            if "outstation_addr" in di: do["outstation_addr"] = int(di["outstation_addr"])
            if "master_addr"     in di: do["master_addr"]     = int(di["master_addr"])
            ch2_out["dnp3"] = do
        payload["ch2"] = ch2_out


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

        # Validate TCP port: must be 502 or in 1000..5000
        try:
            port = int(current_on_disk.get("tcp", {}).get("port", 1502))
        except Exception:
            raise HTTPException(status_code=400, detail="tcp.port must be an integer")

        if not (port == 502 or (1025 <= port <= 5000)):
            raise HTTPException(status_code=400, detail="TCP port must be 502 or between 1000–5000")


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

        global SKIP_NEXT_WATCH_RELOAD
        SKIP_NEXT_WATCH_RELOAD = True



        SETTINGS.clear()
        SETTINGS.update(current_on_disk)

    # ----- after SETTINGS.update(current_on_disk) and before return

    prev_port = (prev.get("tcp", {}) or {}).get("port")
    new_port  = (S().get("tcp", {}) or {}).get("port")
    if prev_port != new_port:
        evt = getattr(app.state, "tcp_reload_event", None)
        if evt:
            evt.set()


    # ----- detect what changed
    prev_mr = (prev.get("mirror_rtu", {}) or {})
    new_mr  = (S().get("mirror_rtu", {}) or {})

    prev_mirror_slave = prev_mr.get("slave_id")
    new_mirror_slave  = new_mr.get("slave_id")

    #if prev_mirror_slave != new_mirror_slave:
        #global PREV_MIRROR_ID, PREV_EXPIRY
        #PREV_MIRROR_ID = prev_mirror_slave
        #PREV_EXPIRY = time.monotonic() + 60  # keep old ID alive for 60s


    # ----- rebuild contexts when hr window / units / mirror slave changed
    need_rebuild = (
        (prev.get("hr", {}) or {}) != (S().get("hr", {}) or {}) or
        (prev.get("local_units", {}) or {}) != (S().get("local_units", {}) or {}) or
        prev_mirror_slave != new_mirror_slave
    )
    if need_rebuild:
        await rebuild_datastores_and_context()

    # after you compute need_rebuild etc., add:
    prev_mode = (prev.get("ch2", {}) or {}).get("mode", "modbus")
    new_mode  = (S().get("ch2",  {}) or {}).get("mode", "modbus")
    mode_changed = (prev_mode != new_mode)

    # serial params changed? (you already have mirror_serial_changed)
    if mirror_serial_changed or mode_changed:
        evt = getattr(app.state, "mirror_reload_event", None)
        if evt: evt.set()


    # ----- only poke the RTU manager if SERIAL parameters changed (NOT slave_id)
    prev_mr = (prev.get("mirror_rtu", {}) or {})
    new_mr  = (S().get("mirror_rtu", {}) or {})
    serial_fields = ("baudrate", "parity", "stopbits", "bytesize")
    mirror_serial_changed = any(prev_mr.get(k) != new_mr.get(k) for k in serial_fields)
    if mirror_serial_changed:
        #mirror_reload_event.set()
        evt = getattr(app.state, "mirror_reload_event", None)
        if evt:
            evt.set()


    return JSONResponse({"ok": True})

"""
from fastapi import HTTPException
from fastapi.responses import JSONResponse

@app.put("/api/settings")
async def put_settings(req: Request, user=Depends(require_admin_or_user)):
    body = await req.json()

    # Load current config (adapt to your storage)
    cfg = load_settings()  # e.g. a dict

    # ---- default flags so we can safely check them later
    mirror_serial_changed: bool = False
    mode_changed: bool = False
    dnp_changed: bool = False

    # ---- Upstream
    if "upstream" in body:
        cfg.setdefault("upstream", {}).update({
            "device_unit_id": int(body["upstream"].get("device_unit_id", cfg["upstream"].get("device_unit_id", 1))),
            "poll_period_s": float(body["upstream"].get("poll_period_s",  cfg["upstream"].get("poll_period_s", 1.0))),
        })

    # ---- Mirror serial (CH2 UART & slave id)
    if "mirror_rtu" in body:
        old_m = cfg.get("mirror_rtu", {})
        new_m = {
            "baudrate": int(body["mirror_rtu"].get("baudrate", old_m.get("baudrate", 9600))),
            "parity":   str(body["mirror_rtu"].get("parity",   old_m.get("parity", "N"))),
            "stopbits": int(body["mirror_rtu"].get("stopbits", old_m.get("stopbits", 1))),
            "bytesize": int(body["mirror_rtu"].get("bytesize", old_m.get("bytesize", 8))),
            "slave_id": int(body["mirror_rtu"].get("slave_id", old_m.get("slave_id", 2))),
        }
        mirror_serial_changed = any(new_m[k] != old_m.get(k) for k in new_m.keys())
        cfg["mirror_rtu"] = {**old_m, **new_m}

    # ---- CH2 mode (modbus/dnp3) + addresses
    if "ch2" in body:
        old_ch2 = cfg.get("ch2", {})
        new_ch2 = {**old_ch2, **body["ch2"]}
        old_mode = (old_ch2.get("mode") or "modbus").lower()
        new_mode = (new_ch2.get("mode") or "modbus").lower()
        mode_changed = (old_mode != new_mode)

        # normalize dnp3 sub-block
        if new_mode == "dnp3":
            old_d = old_ch2.get("dnp3", {}) or {}
            req_d = (body["ch2"].get("dnp3") or {}) if "ch2" in body else {}
            dnp = {
                "outstation_addr": int(req_d.get("outstation_addr", old_d.get("outstation_addr", 100))),
                "master_addr":     int(req_d.get("master_addr",     old_d.get("master_addr", 1))),
            }
            dnp_changed = (dnp["outstation_addr"] != old_d.get("outstation_addr")) or \
                          (dnp["master_addr"]     != old_d.get("master_addr"))
            new_ch2["dnp3"] = dnp

        new_ch2["mode"] = new_mode
        cfg["ch2"] = new_ch2

    # ---- TCP
    if "tcp" in body:
        port = int(body["tcp"].get("port", cfg.get("tcp", {}).get("port", 1502)))
        if not (port == 502 or 1025 <= port <= 5000):
            raise HTTPException(status_code=422, detail="Invalid TCP port")
        cfg.setdefault("tcp", {})["port"] = port

    # ---- Local units (only unit1 is user-editable in your UI)
    if "local_units" in body:
        cfg.setdefault("local_units", {})["unit1_id"] = int(body["local_units"].get("unit1_id", 2))

    # ---- HR window (fixed 0..23 in your UI; keep resilient)
    if "hr" in body:
        cfg["hr"] = {
            "start": int(body["hr"].get("start", 0)),
            "count": int(body["hr"].get("count", 24)),
        }

    # ---- Branding / Device (admin path; ignore if not present)
    if "branding" in body:
        cfg.setdefault("branding", {}).update(body["branding"] or {})
    if "device" in body:
        cfg.setdefault("device", {}).update(body["device"] or {})

    # ---- Persist config
    save_settings(cfg)  # adapt to your code

    # ---- Restart CH2 worker if needed
    if mirror_serial_changed or mode_changed or dnp_changed:
        try:
            restart_ch2_worker(cfg)  # or stop + spawn; adapt to your helpers
        except Exception as e:
            # Don’t fail the whole request; log and still return ok so UI doesn’t spin forever
            print("[CH2] restart failed:", e)

    return JSONResponse({"ok": True})
"""


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

    # create asyncio primitives on the running loop
    global tcp_reload_event, mirror_reload_event, MIRROR_QUEUE
    tcp_reload_event = asyncio.Event()
    mirror_reload_event = asyncio.Event()
    app.state.mirror_reload_event = mirror_reload_event  # let routes access it
    app.state.tcp_reload_event = tcp_reload_event

    MIRROR_QUEUE = asyncio.Queue(maxsize=4)

    # initial stores/context
    await rebuild_datastores_and_context()
    mirror_reload_event.set()

    await asyncio.gather(
        poll_upstream_and_update_cache(),
        tcp_server_manager(),
        ch2_supervisor(),
        start_web(),
        settings_auto_reload(),
    )

if __name__ == "__main__":
    asyncio.run(main())
