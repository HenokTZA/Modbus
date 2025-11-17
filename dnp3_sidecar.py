#!/usr/bin/env python3
"""
DNP3 sidecar for CH2

- Uses Chargebyte / Automatak pydnp3 bindings.
- Listens for JSON lines on stdin: {"op": "snap", "values": [hr0, hr1, ...]}.
- Maps holding registers to DNP3 points:
    - HR[0..count-1] -> Analog Inputs 0..count-1
    - Bits of HR[10] -> Binary Inputs 0..15
    - Bits of HR[11] -> Binary Inputs 16..31
"""

import sys
import json
import argparse
import asyncio
import signal

try:
    # chargebyte fork keeps the same package name
    from pydnp3 import opendnp3, asiopal, asiodnp3
except Exception as e:
    print("[DNP3] pydnp3 not available:", e, file=sys.stderr)
    sys.exit(2)


class SnapshotApplication(opendnp3.IOutstationApplication):
    """
    Minimal outstation application that can apply HR snapshots
    into the outstation database.
    """

    def __init__(self, count: int):
        super().__init__()
        self.count = count
        self._outstation = None  # type: asiodnp3.IOutstation

    def set_outstation(self, outstation):
        self._outstation = outstation

    def apply_snapshot(self, regs):
        """
        regs - list[int] of holding register values (0..65535)
        """
        if self._outstation is None:
            return

        builder = asiodnp3.UpdateBuilder()

        # Map HR -> Analog Inputs
        limit = min(self.count, len(regs))
        for i in range(limit):
            v = float(regs[i] & 0xFFFF)
            builder.Update(opendnp3.Analog(v), i)

        # Bits from HR10 and HR11 -> Binary Inputs 0..31
        if len(regs) > 10:
            a1 = regs[10] & 0xFFFF
            for b in range(16):
                bit = bool((a1 >> b) & 0x1)
                builder.Update(opendnp3.Binary(bit), b)

        if len(regs) > 11:
            a2 = regs[11] & 0xFFFF
            for b in range(16):
                bit = bool((a2 >> b) & 0x1)
                builder.Update(opendnp3.Binary(bit), 16 + b)

        update = builder.Build()
        self._outstation.Apply(update)


class NoopCommandHandler(opendnp3.ICommandHandler):
    """
    Command handler that always returns SUCCESS for any received command.
    You can replace this with real logic later.
    """

    def __init__(self):
        super().__init__()

    def Begin(self):
        pass

    def End(self):
        pass

    def Select(self, command, index, op_type):
        return opendnp3.CommandStatus.SUCCESS

    def Operate(self, command, index, op_type):
        return opendnp3.CommandStatus.SUCCESS

    def DirectOperate(self, command, index, op_type):
        return opendnp3.CommandStatus.SUCCESS

"""
def configure_stack(count: int, outstation_addr: int, master_addr: int):

    # allocate database sizes; make sure we have at least 32 binary inputs
    db_sizes = opendnp3.DatabaseSizes.AllTypes(max(count, 32))
    stack_config = asiodnp3.OutstationStackConfig(db_sizes)

    # basic link config
    stack_config.link.LocalAddr = outstation_addr
    stack_config.link.RemoteAddr = master_addr

    # event buffer
    stack_config.outstation.eventBufferConfig = opendnp3.EventBufferConfig().AllTypes(100)

    # configure database layout
    db = stack_config.dbConfig
    for i in range(count):
        db.analog[i] = opendnp3.AnalogConfig()
    for i in range(32):
        db.binary[i] = opendnp3.BinaryConfig()

    return stack_config
"""

def configure_stack(count, outstation_addr, master_addr):
    """
    Build a simple outstation stack config with default DB settings.

    We just tell it how many points we want via DatabaseSizes.AllTypes,
    and set link-layer addresses. No need to mutate dbConfig arrays.
    """
    # Make sure we have at least 'count' analog points, and some binary headroom.
    db_sizes = opendnp3.DatabaseSizes.AllTypes(max(count, 32))

    # This creates dbConfig with default AnalogConfig/BinaryConfig for all points
    stack_config = asiodnp3.OutstationStackConfig(db_sizes)

    # Link-layer addresses: outstation = local, master = remote
    stack_config.link.LocalAddr = outstation_addr   # e.g. 100
    stack_config.link.RemoteAddr = master_addr      # e.g. 1

    # Optional: tweak event buffer sizes if you want later, but not required now.
    # stack_config.outstation.eventBufferConfig = opendnp3.EventBufferConfig.AllTypes(10)

    return stack_config



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

    # DNP3 manager and serial channel
    log_handler = asiodnp3.ConsoleLogger().Create()
    manager = asiodnp3.DNP3Manager(1, log_handler)

    retry = asiopal.ChannelRetry().Default()





    # Determine the correct "no parity" enum – different builds name it differently
    if hasattr(opendnp3.Parity, "None_"):
        PARITY_NONE = opendnp3.Parity.None_
    elif hasattr(opendnp3.Parity, "None"):
        PARITY_NONE = getattr(opendnp3.Parity, "None")
    else:
        raise RuntimeError("Unsupported pydnp3 Parity enum (no None/None_)")

    parity_map = {
        "N": PARITY_NONE,
        "E": opendnp3.Parity.Even,
        "O": opendnp3.Parity.Odd,
    }

    def set_first_attr(obj, names, value):
        """Set the first existing attribute in 'names' on obj to value."""
        for name in names:
            if hasattr(obj, name):
                setattr(obj, name, value)
                return name
        return None

    # --- SerialSettings: build with default ctor, then fill fields ---
    serial_settings = asiopal.SerialSettings()

    # Device / port name
    set_first_attr(
        serial_settings,
        ("deviceName", "port", "portName", "device"),
        args.port,
    )

    # Baud rate
    set_first_attr(
        serial_settings,
        ("baud", "baudrate"),
        args.baudrate,
    )

    # Data bits
    set_first_attr(
        serial_settings,
        ("dataBits", "data_bits"),
        args.bytesize,
    )

    # Parity
    set_first_attr(
        serial_settings,
        ("parity",),
        parity_map[args.parity.upper()],
    )

    # Stop bits
    stop_enum = opendnp3.StopBits.One if args.stopbits == 1 else opendnp3.StopBits.Two
    set_first_attr(
        serial_settings,
        ("stopBits", "stop_bits"),
        stop_enum,
    )

    # Optional: force no flow control if the attribute exists
    if hasattr(asiopal, "FlowType") and hasattr(serial_settings, "flowType"):
        # Try "None_" first, fall back to "None"
        if hasattr(asiopal.FlowType, "None_"):
            serial_settings.flowType = asiopal.FlowType.None_
        elif hasattr(asiopal.FlowType, "None"):
            serial_settings.flowType = getattr(asiopal.FlowType, "None")


    print("SerialSettings debug:",
          getattr(serial_settings, "deviceName", None),
          getattr(serial_settings, "port", None),
          getattr(serial_settings, "baud", None),
          getattr(serial_settings, "baudrate", None),
          file=sys.stderr)


    # Logging level mask – NORMAL is fine, you can adjust later if needed
    log_levels = opendnp3.levels.NORMAL

    listener = asiodnp3.PrintingChannelListener().Create()

    channel = manager.AddSerial(
        "ch2-serial",
        log_levels,
        retry,
        serial_settings,
        listener,
    )


    # stack config and application
    stack_config = configure_stack(args.count, args.outstation, args.master)
    app = SnapshotApplication(args.count)
    cmd_handler = NoopCommandHandler()

    outstation = channel.AddOutstation(
        "outstation",
        cmd_handler,
        app,
        stack_config,
    )
    app.set_outstation(outstation)

    outstation.Enable()  # start responding

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _sig_handler(*_):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _sig_handler)
        except NotImplementedError:
            # signals are not supported on Windows event loop
            pass

    # set up async reader for stdin (JSON snapshot stream)
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    print(
        f"[DNP3] outstation up on {args.port} baud={args.baudrate} oa={args.outstation} ma={args.master}",
        file=sys.stderr,
    )

    bad_lines = 0

    while not stop_event.is_set():
        line = await reader.readline()
        if not line:
            # EOF or no data; yield to event loop
            await asyncio.sleep(0.05)
            continue

        try:
            msg = json.loads(line.decode().strip())
            if msg.get("op") == "snap":
                regs = [int(x) & 0xFFFF for x in (msg.get("values") or [])]
                app.apply_snapshot(regs)
                bad_lines = 0
        except Exception as e:
            bad_lines += 1
            print("[DNP3] bad line:", e, "raw:", line[:80], file=sys.stderr)
            if bad_lines > 20:
                print("[DNP3] too many bad lines, shutting down", file=sys.stderr)
                break

    outstation.Shutdown()
    channel.Shutdown()
    manager.Shutdown()


if __name__ == "__main__":
    asyncio.run(main())

