# alarms.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import time

@dataclass
class AlarmStatus:
    name: str
    on: bool
    color: str        # "red" or "green"
    msg: str

@dataclass
class AlarmsEngine:
    """
    Keeps small state to support hysteresis (for Abnormal AC mains).
    Everything else is evaluated statically from the latest registers.
    """
    # AC abnormal latch (hysteresis)
    ac_abnormal_on: bool = False

    # indices (based on your ANNEX_A mapping)
    IDX_BATT_V: int = 0
    IDX_LOAD_V: int = 1
    IDX_TOTAL_I: int = 4
    IDX_VRN: int = 5
    IDX_VSN: int = 6
    IDX_VTN: int = 7
    IDX_TEMP: int = 8
    IDX_AL1: int = 10
    IDX_AL2: int = 11

    def _ac_abnormal_eval(self, v_rn: float, v_sn: float, v_tn: float) -> bool:
        """
        Hysteresis:
          - ON if any undervoltage (< 180) OR overvoltage (> 260) OR phase imbalance > 25
          - OFF when all recover to: all in [190, 250] and imbalance < 18
        """
        vmax = max(v_rn, v_sn, v_tn)
        vmin = min(v_rn, v_sn, v_tn)
        imbalance = vmax - vmin

        if not self.ac_abnormal_on:
            # trigger conditions
            if (v_rn < 180 or v_sn < 180 or v_tn < 180) or \
               (v_rn > 260 or v_sn > 260 or v_tn > 260) or \
               (imbalance > 25):
                self.ac_abnormal_on = True
        else:
            # clear conditions
            if (190 <= v_rn <= 250) and (190 <= v_sn <= 250) and (190 <= v_tn <= 250) and (imbalance < 18):
                self.ac_abnormal_on = False

        return self.ac_abnormal_on

    def evaluate(
        self,
        raw_regs: List[int],
        scaled: Dict[str, float],
        last_good_poll_ts: Optional[float],
        poll_period_s: float,
    ) -> List[AlarmStatus]:
        """
        raw_regs: list of 24 raw HR values (0..23)
        scaled: dictionary from /api/hr style (labels -> scaled floats); used where you scale volts/amps/temp
        last_good_poll_ts: monotonic() timestamp of last successful RTU read
        poll_period_s: current poll period (to detect comm loss)
        """
        # convenience getters with sane defaults
        def _raw(idx, default=0):  # raw ints
            return int(raw_regs[idx]) if (0 <= idx < len(raw_regs)) else int(default)

        def _sc(name, default=0.0):  # scaled floats
            return float(scaled.get(name, default))

        # Grab values
        al1, al2 = _raw(self.IDX_AL1), _raw(self.IDX_AL2)
        battery_v = _sc("BATTERY_VOLTAGE_V", 0.0)        # V
        load_v    = _sc("LOAD_VOLTAGE_V", 0.0)           # V
        total_i   = _sc("TOTAL_CURRENT_A", 0.0)          # A
        temp_c    = _sc("AMBIENT_TEMP_C", 0.0)           # °C

        # AC voltages are raw (your ANNEX_A sets scale=1.0); treat them as volts
        v_rn = float(_raw(self.IDX_VRN))
        v_sn = float(_raw(self.IDX_VSN))
        v_tn = float(_raw(self.IDX_VTN))

        # 1) Pole to ground (earth fault): (AL1 & AL2) != 0
        pole_to_ground_on = ((al1 & al2) != 0)

        # 2) High battery voltage (per your spec — note naming seems inverted, we follow your condition):
        #    ON when Battery_voltage_v < 5V
        high_batt_on = (battery_v < 5.0)

        # 3) Low battery voltage (per your spec):
        #    ON when Battery_voltage_v > 10V
        low_batt_on = (battery_v > 10.0)

        # 4) No communication: ON when RTU comm stops
        comm_on = False
        msg_comm = "OK"
        now_mono = time.monotonic()
        if last_good_poll_ts is None:
            comm_on = True
            msg_comm = "No reads yet"
        else:
            # if no successful poll for > 3x poll period, raise
            stale_for = now_mono - last_good_poll_ts
            if stale_for > max(3.0 * max(poll_period_s, 0.1), 1.0):
                comm_on = True
                msg_comm = f"No RTU data for {stale_for:.1f}s"
            else:
                msg_comm = f"Last RTU {stale_for:.1f}s ago"

        # 5) Abnormal AC mains: with hysteresis
        ac_abn_on = self._ac_abnormal_eval(v_rn, v_sn, v_tn)

        # 6) High load/consumption voltage: ON when Vload >= 20, OFF when <= 20
        # (No hysteresis band given, so threshold toggles at 20)
        high_load_on = (load_v >= 20.0)

        # 7) Low load/consumption voltage: ON when Vload <= 3, OFF when >= 3
        low_load_on = (load_v <= 3.0)

        # 8) Fuse open (blown): On when TOTAL_CURRENT_A == 0
        fuse_open_on = (total_i == 0.0)

        # 9) High temperature: when temperature > 26
        high_temp_on = (temp_c > 26.0)

        # Build statuses
        out: List[AlarmStatus] = [
            AlarmStatus("Pole to ground", pole_to_ground_on,
                        "red" if pole_to_ground_on else "green",
                        f"AL1&AL2 = {al1 & al2}"),
            AlarmStatus("High battery voltage", high_batt_on,
                        "red" if high_batt_on else "green",
                        f"Vb={battery_v:.1f}V (<5→ON)"),
            AlarmStatus("Low battery voltage", low_batt_on,
                        "red" if low_batt_on else "green",
                        f"Vb={battery_v:.1f}V (>10→ON)"),
            AlarmStatus("No communication", comm_on,
                        "red" if comm_on else "green",
                        msg_comm),
            AlarmStatus("Abnormal AC mains", ac_abn_on,
                        "red" if ac_abn_on else "green",
                        f"VRN={v_rn:.0f} V, VSN={v_sn:.0f} V, VTN={v_tn:.0f} V"),
            AlarmStatus("High load voltage", high_load_on,
                        "red" if high_load_on else "green",
                        f"Vload={load_v:.1f}V (>=20→ON)"),
            AlarmStatus("Low load voltage", low_load_on,
                        "red" if low_load_on else "green",
                        f"Vload={load_v:.1f}V (<=3→ON)"),
            AlarmStatus("Fuse open (blown)", fuse_open_on,
                        "red" if fuse_open_on else "green",
                        f"Itotal={total_i:.1f}A (==0→ON)"),
            AlarmStatus("High temperature", high_temp_on,
                        "red" if high_temp_on else "green",
                        f"T={temp_c:.1f}°C (>26→ON)"),
        ]
        return out
