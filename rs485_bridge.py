#!/usr/bin/env python3
import os
import pty
import select
import serial
import sys

# Real RS485 HAT device
REAL_PORT = "/dev/ttySC1"
BAUDRATE = 9600

def main():
    # Create a pseudo-terminal pair
    master_fd, slave_fd = pty.openpty()
    slave_path = os.ttyname(slave_fd)

    print("")
    print("=== RS485 ↔ PTY bridge ===")
    print("Real RS485 device :", REAL_PORT)
    print("PTY for DNP3      :", slave_path)
    print("")
    print("Use this PTY path in your dnp3_sidecar/app4 settings as the DNP3 serial port.")
    print("Press Ctrl+C to stop.")
    sys.stdout.flush()

    # Open the real RS485 port with pySerial (we know this works)
    ser = serial.Serial(
        REAL_PORT,
        BAUDRATE,
        bytesize=8,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0,          # non-blocking
    )

    try:
        while True:
            rlist, _, _ = select.select([master_fd, ser.fileno()], [], [])
            # Data from DNP3 (PTY) → RS485
            if master_fd in rlist:
                data = os.read(master_fd, 1024)
                if data:
                    ser.write(data)
            # Data from RS485 → DNP3 (PTY)
            if ser.fileno() in rlist:
                data = ser.read(1024)
                if data:
                    os.write(master_fd, data)
    except KeyboardInterrupt:
        print("\nBridge stopped.")
    finally:
        ser.close()
        os.close(master_fd)
        os.close(slave_fd)


if __name__ == "__main__":
    main()
