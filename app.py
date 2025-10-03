#!/usr/bin/env python3
import asyncio, inspect, time, json, os
from typing import Dict, Tuple, Any

from pymodbus.datastore import (
    ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock
)
from pymodbus.framer import FramerType
from pymodbus.server import StartAsyncSerialServer, StartAsyncTcpServer
from pymodbus.client import AsyncModbusSerialClient

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

SETTINGS_PATH = "settings.json"

# ===================== DEFAULT SETTINGS =====================
DEFAULT_SETTINGS = {
    # Upstream RTU (master->device on CH1)
    "upstream": {
        "port": "/dev/ttySC0",
        "baudrate": 9600,
        "parity": "N",
        "stopbits": 1,
        "bytesize": 8,
        "device_unit_id": 1,
        "poll_period_s": 1.0
    },
    # Downstream RTU mirror (slave on CH2)
    "mirror_rtu": {
        "port": "/dev/ttySC1",
        "baudrate": 9600,
        "parity": "N",
        "stopbits": 1,
        "bytesize": 8
    },
    # TCP slave
    "tcp": {
        "port": 1502
    },
    # Local multi-slave unit IDs
    "local_units": {
        "unit0_id": 1,  # 0-based window (0..23) -> dashboard
        "unit1_id": 2   # 1-based window (1..24) -> Modbus Poll
    },
    # Holding registers mirrored count
    "hr": {
        "start": 0,
        "count": 24
    }
}
# ============================================================

# ---------- Settings helpers ----------
SETTINGS_LOCK = asyncio.Lock()
SETTINGS: Dict[str, Any] = {}

def load_settings_from_disk() -> Dict[str, Any]:
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        # merge defaults to keep compatibility if new fields are added
        def deep_merge(defs, cur):
            if isinstance(defs, dict):
                out = {}
                for k, v in defs.items():
                    if k in cur:
                        out[k] = deep_merge(v, cur[k])
                    else:
                        out[k] = v
                # include any extra keys user added
                for k, v in cur.items():
                    if k not in out:
                        out[k] = v
                return out
            else:
                return cur if cur is not None else defs
        return deep_merge(DEFAULT_SETTINGS, data)
    except Exception:
        return DEFAULT_SETTINGS.copy()

async def save_settings_to_disk(settings: Dict[str, Any]):
    tmp = json.dumps(settings, indent=2)
    with open(SETTINGS_PATH, "w") as f:
        f.write(tmp)

# Load on boot
SETTINGS = load_settings_from_disk()

# Shortcuts
def S(): return SETTINGS  # current settings accessor

# --------- Two datastores (separate, no overlap) ----------
def make_store0(count: int):
    return ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0]),
        co=ModbusSequentialDataBlock(0, [0]),
        hr=ModbusSequentialDataBlock(0, [0] * count),
        ir=ModbusSequentialDataBlock(0, [0])
    )

def make_store1(count: int):
    # +1 so we can write at addr=1 → window 1..count
    return ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0]),
        co=ModbusSequentialDataBlock(0, [0]),
        hr=ModbusSequentialDataBlock(0, [0] * (count + 1)),
        ir=ModbusSequentialDataBlock(0, [0])
    )

HR_COUNT = S()["hr"]["count"]
store0 = make_store0(HR_COUNT)
store1 = make_store1(HR_COUNT)

context = ModbusServerContext(slaves={
    S()["local_units"]["unit0_id"]: store0,
    S()["local_units"]["unit1_id"]: store1
}, single=False)

# A single lock to guard ALL datastore reads/writes
HR_LOCK = asyncio.Lock()

def _hr_block0() -> ModbusSequentialDataBlock:
    return store0.store["h"]

def _hr_block1() -> ModbusSequentialDataBlock:
    return store1.store["h"]

async def _write_both_views(regs):
    """Atomically write into Unit1(0-based @0) and Unit2(1-based @1)."""
    async with HR_LOCK:
        _hr_block0().setValues(0, regs)  # 0..(count-1)
        _hr_block1().setValues(1, regs)  # 1..count

# Scaling meta (unchanged)
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

async def _safe_close(client):
    try:
        if client is None: return
        close_fn = getattr(client, "close", None)
        if close_fn is None: return
        if inspect.iscoroutinefunction(close_fn):
            await close_fn()
        else:
            close_fn()
    except Exception:
        pass

# ------------------ Upstream poller with live RTU reload ------------------
async def poll_upstream_and_update_cache():
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

    async def rtu_settings_changed() -> bool:
        ups = S()["upstream"]
        return any(ups[k] != cur.get(k) for k in ["port","baudrate","parity","stopbits","bytesize"])

    # initial connect
    await connect()

    try:
        while True:
            # Reconnect if RTU settings changed
            if await rtu_settings_changed():
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

# ---------------- Downstream RTU slave ----------------
async def start_downstream_server():
    # NOTE: mirror RTU port/params are read once at startup; change here requires process restart
    mir = S()["mirror_rtu"]
    await StartAsyncSerialServer(
        context=context,
        framer=FramerType.RTU,
        port=mir["port"],
        baudrate=mir["baudrate"],
        parity=mir["parity"],
        stopbits=mir["stopbits"],
        bytesize=mir["bytesize"],
        timeout=1
    )

# -------------------- TCP slave ----------------------
async def start_tcp_server():
    # NOTE: TCP port is read once at startup; change requires process restart
    await StartAsyncTcpServer(
        context=context,
        address=("0.0.0.0", S()["tcp"]["port"]),
        ignore_missing_slaves=True,
    )

# ---------------- Web API & Root Dashboard -----------
app = FastAPI()

@app.get("/api/hr")
async def api_hr():
    # Dashboard uses the Unit 1 (0-based) values at 0..(count-1)
    async with HR_LOCK:
        regs = _hr_block0().getValues(0, S()["hr"]["count"])
    out: Dict[str, Any] = {
        "raw": regs,
        "scaled": {},
        "meta": {"unit_id": S()["local_units"]["unit0_id"], "start": 0, "count": S()["hr"]["count"]}
    }
    for i, _v in enumerate(regs):
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

# -------- Settings API --------
@app.get("/api/settings")
async def get_settings():
    async with SETTINGS_LOCK:
        return JSONResponse(S())

@app.put("/api/settings")
async def put_settings(payload: Dict[str, Any] = Body(...)):
    # Basic validation/merge
    async with SETTINGS_LOCK:
        new = load_settings_from_disk()  # start from current-on-disk merged to defaults
        def deep_merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    deep_merge(dst[k], v)
                else:
                    dst[k] = v
        deep_merge(new, payload)

        # persist
        try:
            await save_settings_to_disk(new)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")

        # update in-memory
        SETTINGS.clear()
        SETTINGS.update(new)

    return JSONResponse({"ok": True, "note":
        "Upstream RTU changes apply live; TCP port & mirror RTU changes require restarting the process."})

# Serve external dashboard.html at root
@app.get("/")
async def root():
    return FileResponse("dashboard.html", media_type="text/html")

# --------------------- Server boot -------------------
async def start_web():
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    await asyncio.gather(
        poll_upstream_and_update_cache(),
        start_downstream_server(),
        start_tcp_server(),
        start_web(),
    )

if __name__ == "__main__":
    asyncio.run(main())
