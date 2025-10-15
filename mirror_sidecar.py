#!/usr/bin/env python3
import asyncio, json, sys, signal, argparse, contextlib
from typing import List

from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock
from pymodbus.framer import FramerType
from pymodbus.server import StartAsyncSerialServer

# Protocol over stdin (one JSON per line):
#   {"op":"snap","values":[<int>...]}  -> set full HR view at address 1
#   {"op":"set","start":X,"values":[...]} -> set partial HR starting at addr X (1-based)
#   {"op":"ping"} -> respond {"op":"pong"}

def make_store1(count: int) -> ModbusSlaveContext:
    # 1-based view (addr 1..count populated)
    return ModbusSlaveContext(
        di=ModbusSequentialDataBlock(0, [0]),
        co=ModbusSequentialDataBlock(0, [0]),
        hr=ModbusSequentialDataBlock(0, [0] * (count + 1)),
        ir=ModbusSequentialDataBlock(0, [0]),
    )

async def stdin_loop(store: ModbusSequentialDataBlock):
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else line)
        except Exception:
            continue

        op = msg.get("op")
        if op == "ping":
            with contextlib.suppress(Exception):
                sys.stdout.write(json.dumps({"op": "pong"}) + "\n")
                sys.stdout.flush()
            continue

        if op == "snap":
            vals: List[int] = list(map(int, msg.get("values", [])))
            store.setValues(1, vals)  # 1-based
            continue

        if op == "set":
            start = int(msg.get("start", 1))
            vals: List[int] = list(map(int, msg.get("values", [])))
            store.setValues(start, vals)
            continue

async def run_server(args):
    store = make_store1(args.count)
    ctx = ModbusServerContext(slaves={args.slave_id: store}, single=False)

    async def _server():
        await StartAsyncSerialServer(
            context=ctx,
            framer=FramerType.RTU,
            port=args.port,
            baudrate=args.baudrate,
            parity=args.parity.upper()[:1],
            stopbits=args.stopbits,
            bytesize=args.bytesize,
            timeout=1,
            ignore_missing_slaves=True,
        )

    stop = asyncio.Event()
    def _signal(*_): stop.set()
    signal.signal(signal.SIGTERM, _signal)
    signal.signal(signal.SIGINT,  _signal)

    server_task = asyncio.create_task(_server(), name="mbserial")
    stdin_task  = asyncio.create_task(stdin_loop(store.store["h"]), name="stdin-loop")

    done, pending = await asyncio.wait(
        {server_task, stdin_task, stop.wait()},
        return_when=asyncio.FIRST_COMPLETED
    )
    for t in (server_task, stdin_task):
        with contextlib.suppress(Exception):
            t.cancel()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--baudrate", type=int, required=True)
    p.add_argument("--parity", default="N")
    p.add_argument("--stopbits", type=int, default=1)
    p.add_argument("--bytesize", type=int, default=8)
    p.add_argument("--slave-id", type=int, required=True)
    p.add_argument("--count", type=int, required=True)  # HR count
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_server(args))
