# alarms.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
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
            if (v_rn < 180 or v_sn < 180 or v_tn < 180) or \
               (v_rn > 260 or v_sn > 260 or v_tn > 260) or \
               (imbalance > 25):
                self.ac_abnormal_on = True
        else:
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
        scaled: dictionary from /api/hr style (labels -> scaled floats)
        last_good_poll_ts: monotonic() timestamp of last successful RTU read
        poll_period_s: current poll period (to detect comm loss)
        """
        def _raw(idx, default=0):
            return int(raw_regs[idx]) if (0 <= idx < len(raw_regs)) else int(default)

        def _sc(name, default=0.0):
            return float(scaled.get(name, default))

        al1, al2 = _raw(self.IDX_AL1), _raw(self.IDX_AL2)
        battery_v = _sc("BATTERY_VOLTAGE_V", 0.0)
        load_v    = _sc("LOAD_VOLTAGE_V", 0.0)
        total_i   = _sc("TOTAL_CURRENT_A", 0.0)
        temp_c    = _sc("AMBIENT_TEMP_C", 0.0)

        # AC voltages treated as volts (ANNEX_A scale=1.0)
        v_rn = float(_raw(self.IDX_VRN))
        v_sn = float(_raw(self.IDX_VSN))
        v_tn = float(_raw(self.IDX_VTN))

        # 1) Pole to ground (earth fault): (AL1 & AL2) != 0
        pole_to_ground_on = ((al1 & al2) != 0)

        # 2) High battery voltage: ON when Vb > 10 V
        high_batt_on = (battery_v > 10.0)

        # 3) Low battery voltage: ON when Vb < 5 V
        low_batt_on = (battery_v < 5.0)

        # 4) No communication
        comm_on = False
        msg_comm = "OK"
        now_mono = time.monotonic()
        if last_good_poll_ts is None:
            comm_on = True
            msg_comm = "No reads yet"
        else:
            stale_for = now_mono - last_good_poll_ts
            if stale_for > max(3.0 * max(poll_period_s, 0.1), 1.0):
                comm_on = True
                msg_comm = f"No RTU data for {stale_for:.1f}s"
            else:
                msg_comm = f"Last RTU {stale_for:.1f}s ago"

        # 5) Abnormal AC mains (with hysteresis)
        ac_abn_on = self._ac_abnormal_eval(v_rn, v_sn, v_tn)

        # 6) High load voltage: ON when Vload >= 20
        high_load_on = (load_v >= 20.0)

        # 7) Low load voltage: ON when Vload <= 3
        low_load_on = (load_v <= 3.0)

        # 8) Fuse open (blown): On when Itotal == 0
        fuse_open_on = (total_i == 0.0)

        # 9) High temperature: > 26 °C
        high_temp_on = (temp_c > 26.0)

        return [
            AlarmStatus("Pole to ground", pole_to_ground_on,
                        "red" if pole_to_ground_on else "green",
                        f"AL1&AL2 = {al1 & al2}"),
            AlarmStatus("High battery voltage", high_batt_on,
                        "red" if high_batt_on else "green",
                        f"Vb={battery_v:.1f}V (>10→ON)"),
            AlarmStatus("Low battery voltage", low_batt_on,
                        "red" if low_batt_on else "green",
                        f"Vb={battery_v:.1f}V (<5→ON)"),
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
