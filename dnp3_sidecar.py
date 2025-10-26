#!/usr/bin/env python3
import sys, json, argparse, asyncio, signal

# DNP3 python bindings (pip install pydnp3)
try:
    from pydnp3 import asiodnp3, asiopal, opendnp3
except Exception as e:
    print("[DNP3] pydnp3 not available:", e, file=sys.stderr)
    sys.exit(2)

def make_db(count):
    # Map the first N holding registers to Analog Inputs (AI)
    cfg = opendnp3.DatabaseConfig()
    for i in range(count):
        cfg.analog_input[i] = opendnp3.AnalogInputConfig()
    # Optional: a few Binary Inputs for alarm bits (from HR10/HR11)
    for i in range(32):
        cfg.binary[i] = opendnp3.BinaryConfig()
    return cfg

class SnapshotHandler(opendnp3.IOutstationApplication, opendnp3.ICommandHandler):
    def __init__(self, count, stack):
        super().__init__()
        self.count = count
        self.stack = stack

    # ICommandHandler noop
    def Begin(self): pass
    def End(self): pass
    def Select(self, command, index, opType):  return opendnp3.CommandStatus.SUCCESS
    def Operate(self, command, index, opType): return opendnp3.CommandStatus.SUCCESS
    def DirectOperate(self, command, index, opType): return opendnp3.CommandStatus.SUCCESS

    def apply_snapshot(self, regs):
        updates = opendnp3.UpdateBuilder()
        # HR -> Analog Inputs
        for i in range(min(self.count, len(regs))):
            updates.Update(opendnp3.Analog(regs[i]), i)
        # Bits from HR10/11 -> Binary inputs 0..15 and 16..31
        if len(regs) > 10:
            a1 = regs[10] & 0xFFFF
            for b in range(16):
                updates.Update(opendnp3.Binary(bool((a1>>b)&1)), b)
        if len(regs) > 11:
            a2 = regs[11] & 0xFFFF
            for b in range(16):
                updates.Update(opendnp3.Binary(bool((a2>>b)&1)), 16+b)
        self.stack.SetUpdateHandler().Apply(updates.Build())

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baudrate", type=int, default=9600)
    ap.add_argument("--parity", default="N")
    ap.add_argument("--stopbits", type=int, default=1)
    ap.add_argument("--bytesize", type=int, default=8)
    ap.add_argument("--outstation", type=int, default=100)
    ap.add_argument("--master", type=int, default=1)
    ap.add_argument("--count", type=int, default=24)
    args = ap.parse_args()

    # DNP3 manager
    mgr = asiodnp3.DNP3Manager(1)
    channel = mgr.AddSerial(
        "ch2-serial",
        asiopal.ChannelRetry.Default(),
        asiopal.SerialSettings(
            args.port, args.baudrate, args.dataBits if hasattr(args,'dataBits') else args.bytesize,
            {'N': opendnp3.Parity.None_, 'E': opendnp3.Parity.Even, 'O': opendnp3.Parity.Odd}[args.parity.upper()],
            opendnp3.StopBits.One if args.stopbits==1 else opendnp3.StopBits.Two,
            0, 0
        ),
        asiodnp3.PrintingChannelListener.Create()
    )

    cfg = asiodnp3.OutstationStackConfig(
        make_db(args.count),
        opendnp3.DefaultOutstationApplication(),
        asiodnp3.DefaultMasterApplication()
    )
    # link-layer addresses
    cfg.link.LocalAddr  = args.outstation
    cfg.link.RemoteAddr = args.master
    cfg.outstation.eventBufferConfig = opendnp3.EventBufferConfig.AllTypes(100)

    out_app = SnapshotHandler(args.count, None)  # we’ll set after AddOutstation
    outstation = channel.AddOutstation(
        "out",
        asiodnp3.PrintingSOEHandler.Create(),
        out_app,
        asiodnp3.DefaultCommandHandler.Create(),
        cfg
    )
    out_app.stack = outstation
    outstation.Enable()  # start serving

    loop = asyncio.get_event_loop()
    stop = asyncio.Event()

    def _sig(*_):
        stop.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(s, _sig)
        except NotImplementedError: pass

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    print(f"[DNP3] outstation up on {args.port} (baud {args.baudrate}) oa={args.outstation} ma={args.master}", file=sys.stderr)

    while not stop.is_set():
        line = await reader.readline()
        if not line:
            await asyncio.sleep(0.05)
            continue
        try:
            msg = json.loads(line.decode().strip())
            if msg.get("op") == "snap":
                regs = [int(x) & 0xFFFF for x in (msg.get("values") or [])]
                out_app.apply_snapshot(regs)
        except Exception as e:
            print("[DNP3] bad line:", e, file=sys.stderr)

    outstation.Shutdown()
    channel.Shutdown()
    mgr.Shutdown()

if __name__ == "__main__":
    asyncio.run(main())
