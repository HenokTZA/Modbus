#!/usr/bin/env python3
import asyncio, inspect, time, json, os, contextlib
from typing import Dict, Tuple, Any, List

from pymodbus.datastore import (
    ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock
)
from pymodbus.framer import FramerType
from pymodbus.server import StartAsyncSerialServer, StartAsyncTcpServer
from pymodbus.client import AsyncModbusSerialClient

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

from alarms import AlarmsEngine
import time


from fastapi.staticfiles import StaticFiles
from pathlib import Path


app = FastAPI()

# serve ./static at /static (use absolute path so it works no matter the cwd)
app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "static")),
    name="static",
)

ALARM_ENGINE = AlarmsEngine()
LAST_GOOD_POLL_MONO = None  # monotonic() timestamp of last successful upstream poll


# ===================== SETTINGS =====================
SETTINGS_PATH = "settings.json"

DEFAULT_SETTINGS = {
    "upstream": {                 # master -> device on CH1
        "port": "/dev/ttySC0",
        "baudrate": 9600,
        "parity": "N",
        "stopbits": 1,
        "bytesize": 8,
        "device_unit_id": 1,
        "poll_period_s": 1.0
    },
    "mirror_rtu": {               # slave on CH2 (hot-reload supported)
        "port": "/dev/ttySC1",
        "baudrate": 9600,
        "parity": "N",
        "stopbits": 1,
        "bytesize": 8
    },
    "tcp": {                      # Modbus TCP (hot-reload supported)
        "port": 1502
    },
    "local_units": {              # two views, both hot-reloadable
        "unit0_id": 1,           # 0-based @ 0..(count-1) -> dashboard
        "unit1_id": 2            # 1-based @ 1..count      -> Modbus Poll
    },
    "hr": {                       # mirrored HR window (hot-reloadable)
        "start": 0,
        "count": 24
    },

    "branding": {
        "phone": "011-4639-8310",
        "email": "info@adaxtecna.com",
        "youtube": "https://www.youtube.com/@adaxtecna",
        "logo_url": "/static/adax_logo.png",             # e.g. "/static/adax_logo.png" (optional)
        "qr_url": ""                # If empty, the dashboard will auto-generate from youtube link
    },
    "device": {
        "model": ""                 # user-entered, e.g. "CEA-24/20"
    }
    # keep existing "hr", "tcp", "upstream", etc.

}
# =====================================================

# ---------- Settings helpers ----------
SETTINGS_LOCK = asyncio.Lock()
SETTINGS: Dict[str, Any] = {}

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
HR_LOCK = asyncio.Lock()  # guards HR reads/writes
store0: ModbusSlaveContext = None
store1: ModbusSlaveContext = None
context: ModbusServerContext = None

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
    # safe setValues wrapper for a list
    dst.setValues(start_addr, src)

async def rebuild_datastores_and_context():
    """Rebuilds stores/context if Unit IDs or HR sizing changed; preserves values where possible."""
    global store0, store1, context
    hr_start = S()["hr"]["start"]
    hr_count = S()["hr"]["count"]
    u0 = S()["local_units"]["unit0_id"]
    u1 = S()["local_units"]["unit1_id"]

    # snapshot old data (if exists)
    old0 = []
    old1 = []
    if store0 and store1:
        with contextlib.suppress(Exception):
            old0 = _hr_block0().getValues(0, hr_count)  # might be different size; ok to trim later
        with contextlib.suppress(Exception):
            old1 = _hr_block1().getValues(1, hr_count)

    # build new stores
    new0 = make_store0(hr_count)
    new1 = make_store1(hr_count)

    # copy overlapping data
    if old0:
        _copy_hr_values(old0[:hr_count], new0.store["h"], 0)
    if old1:
        _copy_hr_values(old1[:hr_count], new1.store["h"], 1)

    # swap under lock
    async with HR_LOCK:
        store0 = new0
        store1 = new1
        context = ModbusServerContext(slaves={u0: store0, u1: store1}, single=False)

async def _write_both_views(regs: List[int]):
    async with HR_LOCK:
        _hr_block0().setValues(0, regs)  # 0..count-1
        _hr_block1().setValues(1, regs)  # 1..count

# --------------- Scaling meta (unchanged) ---------------
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
        cur = ups.copy()
        client = AsyncModbusSerialClient(
            port=ups["port"],
            framer=FramerType.RTU,
            baudrate=ups["baudrate"],
            parity=ups["parity"],
            stopbits=ups["stopbits"],
            bytesize=ups["bytesize"],
            timeout=2
        )
        ok = await client.connect()
        if not ok:
            await _safe_close(client)
            client = None
            raise RuntimeError(f"Failed to open upstream port {ups['port']}")

    def rtu_settings_changed() -> bool:
        ups = S()["upstream"]
        return any(ups[k] != cur.get(k) for k in ["port","baudrate","parity","stopbits","bytesize"])

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

                # Logs (best-effort)
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


"""
async def tcp_server_manager():

    current_port = None
    server_task: asyncio.Task = None

    async def start_server(port: int):
        async def run():
            await StartAsyncTcpServer(
                context=context,
                address=("0.0.0.0", port),
                ignore_missing_slaves=True,
            )
        return asyncio.create_task(run())

    try:
        while True:
            desired_port = S()["tcp"]["port"]
            if desired_port != current_port or (server_task and server_task.done()):
                # (Re)start
                if server_task and not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError): await server_task
                current_port = desired_port
                server_task = start_server(current_port)
                print(f"[TCP] listening on 0.0.0.0:{current_port}")

            # wait for reload signal or small sleep
            try:
                await asyncio.wait_for(tcp_reload_event.wait(), timeout=1.0)
                tcp_reload_event.clear()
            except asyncio.TimeoutError:
                pass
    finally:
        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError): await server_task

async def mirror_rtu_server_manager():

    current = {}
    server_task: asyncio.Task = None

    async def start_server(cfg):
        async def run():
            await StartAsyncSerialServer(
                context=context,
                framer=FramerType.RTU,
                port=cfg["port"],
                baudrate=cfg["baudrate"],
                parity=cfg["parity"],
                stopbits=cfg["stopbits"],
                bytesize=cfg["bytesize"],
                timeout=1
            )
        return asyncio.create_task(run())

    try:
        while True:
            cfg = S()["mirror_rtu"]
            changed = any(cfg.get(k) != current.get(k) for k in ["port","baudrate","parity","stopbits","bytesize"])
            if changed or server_task is None or server_task.done():
                if server_task and not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError): await server_task
                current = cfg.copy()
                server_task = start_server(current)
                print(f"[RTU mirror] {current['port']} {current['baudrate']} {current['parity']} {current['stopbits']} {current['bytesize']}")

            try:
                await asyncio.wait_for(mirror_reload_event.wait(), timeout=1.0)
                mirror_reload_event.clear()
            except asyncio.TimeoutError:
                pass
    finally:
        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError): await server_task

"""


async def tcp_server_manager():
    """Hot-reloadable Modbus TCP server (polls settings & restarts when needed)."""
    current_port = None
    server_task: asyncio.Task | None = None

    async def _start(port: int) -> asyncio.Task:
        async def run():
            await StartAsyncTcpServer(
                context=context,
                address=("0.0.0.0", port),
                ignore_missing_slaves=True,
            )
        return asyncio.create_task(run(), name=f"mbtcp:{port}")

    try:
        while True:
            desired_port = S()["tcp"]["port"]
            needs_restart = (
                desired_port != current_port or
                (server_task is not None and server_task.done())
            )

            if needs_restart:
                # stop old
                if server_task and not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task

                # start new
                current_port = desired_port
                server_task = await _start(current_port)
                print(f"[TCP] listening on 0.0.0.0:{current_port}")

            await asyncio.sleep(0.5)  # poll settings
    finally:
        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task


async def mirror_rtu_server_manager():
    """Hot-reloadable Modbus RTU mirror on CH2 (polls settings & restarts when needed)."""
    current = {}
    server_task: asyncio.Task | None = None

    async def _start(cfg: dict) -> asyncio.Task:
        async def run():
            await StartAsyncSerialServer(
                context=context,
                framer=FramerType.RTU,
                port=cfg["port"],
                baudrate=cfg["baudrate"],
                parity=cfg["parity"],
                stopbits=cfg["stopbits"],
                bytesize=cfg["bytesize"],
                timeout=1
            )
        return asyncio.create_task(run(), name=f"mbserial:{cfg['port']}")

    try:
        while True:
            cfg = S()["mirror_rtu"]
            changed = any(cfg.get(k) != current.get(k) for k in ["port","baudrate","parity","stopbits","bytesize"])
            needs_restart = changed or (server_task is None) or server_task.done()

            if needs_restart:
                if server_task and not server_task.done():
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task

                current = cfg.copy()
                server_task = await _start(current)
                print(f"[RTU mirror] {current['port']} {current['baudrate']} {current['parity']} {current['stopbits']} {current['bytesize']}")

            await asyncio.sleep(0.5)  # poll settings
    finally:
        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task




# ================== Web API & Dashboard ==================
#app = FastAPI()


@app.get("/api/alarms")
async def api_alarms():
    # Get the latest raw regs (0-based window for dashboard)
    async with HR_LOCK:
        regs = _hr_block0().getValues(0, S()["hr"]["count"])

    # Compose a 'scaled' dict similar to /api/hr to reuse labels
    scaled = {}
    for i, raw in enumerate(regs):
        label = ANNEX_A.get(i, (f"HR{i}", 1.0))[0]
        div   = ANNEX_A.get(i, ("", 1.0))[1]
        scaled[label] = (raw / div) if div and div != 1.0 else float(raw)

    # Evaluate alarms
    ups = S()["upstream"]
    statuses = ALARM_ENGINE.evaluate(
        raw_regs=regs,
        scaled=scaled,
        last_good_poll_ts=LAST_GOOD_POLL_MONO,
        poll_period_s=float(ups.get("poll_period_s", 1.0)),
    )

    # Build JSON payload
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
        return JSONResponse(S())
"""
@app.put("/api/settings")
async def put_settings(payload: Dict[str, Any] = Body(...)):
    # Merge & save
    async with SETTINGS_LOCK:
        current = load_settings_from_disk()
        def deep_merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v
        deep_merge(current, payload)
        try:
            await save_settings_to_disk(current)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

        # detect what changed
        prev = SETTINGS.copy()
        SETTINGS.clear()
        SETTINGS.update(current)

    # Hot actions:
    # 1) If HR window or local unit IDs changed -> rebuild stores/context
    need_rebuild = (
        prev.get("hr", {}) != S().get("hr", {}) or
        prev.get("local_units", {}) != S().get("local_units", {})
    )
    if need_rebuild:
        await rebuild_datastores_and_context()
        # Nudge servers to pick up new context map
        tcp_reload_event.set()
        mirror_reload_event.set()

    # 2) If TCP changed -> nudge TCP manager
    if prev.get("tcp", {}) != S().get("tcp", {}):
        tcp_reload_event.set()

    # 3) If mirror RTU changed -> nudge mirror manager
    if prev.get("mirror_rtu", {}) != S().get("mirror_rtu", {}):
        mirror_reload_event.set()

    # 4) Upstream RTU is handled live in the poller

    return JSONResponse({"ok": True})
"""

@app.put("/api/settings")
async def put_settings(payload: Dict[str, Any] = Body(...)):
    # Merge & save
    async with SETTINGS_LOCK:
        current_on_disk = load_settings_from_disk()
        def deep_merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v
        deep_merge(current_on_disk, payload)

        try:
            await save_settings_to_disk(current_on_disk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

        prev = SETTINGS.copy()
        SETTINGS.clear()
        SETTINGS.update(current_on_disk)

    # If HR window or local unit IDs changed → rebuild stores/context immediately.
    need_rebuild = (
        prev.get("hr", {}) != S().get("hr", {}) or
        prev.get("local_units", {}) != S().get("local_units", {})
    )
    if need_rebuild:
        await rebuild_datastores_and_context()
        # Managers poll S() and will pick up the new context map automatically.

    # No event.set() calls needed — managers poll every 0.5s and restart on change.
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
