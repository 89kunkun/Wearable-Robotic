# state.py
import math
import threading
from enum import Enum, auto
from typing import Tuple, Optional


def quat_to_euler(w: float, x: float, y: float, z: float):
    """
    Convert quaternion into Euler angles (yaw, pitch, roll).
    Returned units: radians.
    Rotation convention:
        - yaw   : rotation around Z-axis
        - pitch : rotation around Y-axis
        - roll  : rotation around X-axis
    """
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # Out-of-range values are clamped to ±90°
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw, pitch, roll


def rad2deg(r: float) -> float:
    """Convert radians to degrees."""
    return r * 180.0 / math.pi


def quat_conjugate(w: float, x: float, y: float, z: float):
    """
    Conjugate of a unit quaternion.
    For unit quaternions, the conjugate is also the inverse rotation.
    """
    return (w, -x, -y, -z)


def quat_multiply(w1: float, x1: float, y1: float, z1: float,
                  w2: float, x2: float, y2: float, z2: float):
    """
    Quaternion multiplication: q_out = q1 * q2
    Represents composition of rotations (first q2, then q1).
    """
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return w, x, y, z


# -------------------------------------------------------------------
# Discrete direction classification for yaw / pitch / roll
# -------------------------------------------------------------------
class AxisDir(Enum):
    NEG = auto()    # e.g. left / down / ccw
    ZERO = auto()   # still
    POS = auto()    # e.g. right / up / cw


def classify_axis(
    angle_deg: float,
    prev_state: Optional[AxisDir],
    neg_label: str,
    zero_label: str,
    pos_label: str,
    th_low: float = 15.0,
    th_high: float = 30.0,
) -> Tuple[str, AxisDir]:
    """
    Classify an angle (in degrees) into a discrete direction with hysteresis.

    Zones:
        angle <= -th_high             -> NEG  (e.g. "left")
        -th_low <= angle <= th_low    -> ZERO (e.g. "still")
        angle >=  th_high             -> POS  (e.g. "right")
    """
    a = angle_deg

    if a <= -th_high:
        state = AxisDir.NEG
    elif a >= th_high:
        state = AxisDir.POS
    elif -th_low <= a <= th_low:
        state = AxisDir.ZERO
    else:
        # Transition region: keep previous state if we have one
        if prev_state is not None:
            state = prev_state
        else:
            state = AxisDir.ZERO

    if state == AxisDir.NEG:
        label = neg_label
    elif state == AxisDir.POS:
        label = pos_label
    else:
        label = zero_label

    return label, state


class IMUState:
    """
    Maintains the current IMU quaternion state and computes:
      - absolute yaw/pitch/roll (optional)
      - relative yaw/pitch/roll (based on a zero orientation)
      - discrete direction labels for yaw/pitch/roll
      - an 'active' flag derived from relative pitch (arm up/down)
    """

    def __init__(self) -> None:
        # Current quaternion (from IMU)
        self._w: float = 1.0
        self._x: float = 0.0
        self._y: float = 0.0
        self._z: float = 0.0

        # Zero-reference quaternion (arm down)
        self._w0: float = 1.0
        self._x0: float = 0.0
        self._y0: float = 0.0
        self._z0: float = 0.0
        self.zero_calibrated: bool = False

        # Absolute Euler angles in degrees (from current quaternion)
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self.roll_deg: float = 0.0

        # Relative Euler angles in degrees (from q_rel = q * q0^-1)
        self.yaw_rel_deg: float = 0.0
        self.pitch_rel_deg: float = 0.0
        self.roll_rel_deg: float = 0.0

        # Reference for direction (set when active becomes True)
        self._dir_ref_yaw_deg: float = 0.0
        self._dir_ref_pitch_deg: float = 0.0
        self._dir_ref_roll_deg: float = 0.0
        self._dir_ref_valid: bool = False

        # Discrete direction states (internal) and public labels
        self._yaw_dir_state: Optional[AxisDir] = None
        self._pitch_dir_state: Optional[AxisDir] = None
        self._roll_dir_state: Optional[AxisDir] = None

        self.yaw_dir: str = "still"    # "left" / "still" / "right"
        self.pitch_dir: str = "still"  # "down" / "still" / "up"
        self.roll_dir: str = "still"   # "ccw" / "still" / "cw"

        # Activation flag: True when arm is raised above threshold
        self.active: bool = False

        self._lock = threading.Lock()

    def update_from_line(self, line: str) -> None:
        """
        Parse a line of the form: "w,x,y,z"
        Update the internal quaternion state, compute:
          - absolute Euler angles
          - relative Euler angles (if zero calibrated)
          - activation state based on pitch_rel_deg
          - discrete direction labels (based on deltas from activation pose)
        """
        try:
            parts = line.strip().split(",")
            if len(parts) != 4:
                return

            w, x, y, z = map(float, parts)
        except ValueError:
            # Ignore malformed lines
            return

        with self._lock:
            # Update current quaternion
            self._w, self._x, self._y, self._z = w, x, y, z

            # Absolute Euler angles (optional, mainly for debugging)
            yaw_abs, pitch_abs, roll_abs = quat_to_euler(w, x, y, z)
            self.yaw_deg = rad2deg(yaw_abs)
            self.pitch_deg = rad2deg(pitch_abs)
            self.roll_deg = rad2deg(roll_abs)

            # If zero is not calibrated yet, use this orientation as zero
            if not self.zero_calibrated:
                self._w0, self._x0, self._y0, self._z0 = w, x, y, z
                self.zero_calibrated = True
                print("Zero orientation calibrated (arm down).")
                # No relative angles yet, return here
                return

            # Compute relative quaternion: q_rel = q_current * q0_conjugate
            w0c, x0c, y0c, z0c = quat_conjugate(self._w0, self._x0, self._y0, self._z0)
            wr, xr, yr, zr = quat_multiply(w, x, y, z, w0c, x0c, y0c, z0c)

            # Normalize q_rel to avoid drift from numerical error
            norm = math.sqrt(wr * wr + xr * xr + yr * yr + zr * zr)
            if norm > 0.0:
                wr /= norm
                xr /= norm
                yr /= norm
                zr /= norm

            # Relative Euler angles
            yaw_rel, pitch_rel, roll_rel = quat_to_euler(wr, xr, yr, zr)
            self.yaw_rel_deg = rad2deg(yaw_rel)
            self.pitch_rel_deg = rad2deg(pitch_rel)
            self.roll_rel_deg = rad2deg(roll_rel)

            # Update activation state based on relative pitch
            self._update_activation_locked()

            # Update discrete direction labels for yaw/pitch/roll
            self._update_directions_locked()

            # Debug print
            print(
                f"[ABS] Yaw={self.yaw_deg:7.2f}°, "
                f"Pitch={self.pitch_deg:7.2f}°, "
                f"Roll={self.roll_deg:7.2f}°  |  "
                f"[REL] Yaw={self.yaw_rel_deg:7.2f}°, "
                f"Pitch={self.pitch_rel_deg:7.2f}°, "
                f"Roll={self.roll_rel_deg:7.2f}°  |  "
                f"YawDir={self.yaw_dir:>5}, "
                f"PitchDir={self.pitch_dir:>5}, "
                f"RollDir={self.roll_dir:>5}  |  "
                f"Active={self.active}"
            )

    def _update_activation_locked(self) -> None:
        """
        Decide whether IMU control is active based on relative pitch.
        Uses a simple hysteresis:
          - pitch_rel_deg > TH_ON  -> activate
          - pitch_rel_deg < TH_OFF -> deactivate

        When we switch from inactive -> active, we also record the current
        relative Euler angles as the reference for direction classification.
        """
        TH_ON = 20.0   # degrees: arm raised above this → activate
        TH_OFF = 10.0  # degrees: arm lowered below this → deactivate

        if not self.active:
            if self.pitch_rel_deg > TH_ON:
                self.active = True
                # Set direction reference at activation moment
                self._dir_ref_yaw_deg = self.yaw_rel_deg
                self._dir_ref_pitch_deg = self.pitch_rel_deg
                self._dir_ref_roll_deg = self.roll_rel_deg
                self._dir_ref_valid = True
                print("IMU control ACTIVATED (arm up, direction ref set).")
        else:
            if self.pitch_rel_deg < TH_OFF:
                self.active = False
                # When deactivated, reference is no longer used
                self._dir_ref_valid = False
                print("IMU control DEACTIVATED (arm down).")

    def _update_directions_locked(self) -> None:
        """
        Update discrete direction labels (yaw_dir, pitch_dir, roll_dir)
        based on relative yaw/pitch/roll with hysteresis.

        IMPORTANT:
        - If we have a valid direction reference (IMU is active and ref set),
          we use angle_delta = (current_rel - ref_rel).
        - This means: at the moment of activation, all directions are "still".
          Only further movements away from that pose will change the labels.
        """
        # Use delta from activation reference if available
        if self._dir_ref_valid:
            dyaw = self.yaw_rel_deg - self._dir_ref_yaw_deg
            dpitch = self.pitch_rel_deg - self._dir_ref_pitch_deg
            droll = self.roll_rel_deg - self._dir_ref_roll_deg
        else:
            # If not active or no ref, fall back to direct relative angles
            dyaw = self.yaw_rel_deg
            dpitch = self.pitch_rel_deg
            droll = self.roll_rel_deg

        # Yaw: left / still / right
        self.yaw_dir, self._yaw_dir_state = classify_axis(
            angle_deg=dyaw,
            prev_state=self._yaw_dir_state,
            neg_label="left",
            zero_label="still",
            pos_label="right",
            th_low=15.0,
            th_high=30.0,
        )

        # Pitch: down / still / up
        self.pitch_dir, self._pitch_dir_state = classify_axis(
            angle_deg=dpitch,
            prev_state=self._pitch_dir_state,
            neg_label="down",   # or "backward"
            zero_label="still",
            pos_label="up",     # or "forward"
            th_low=15.0,
            th_high=30.0,
        )

        # Roll: ccw / still / cw
        self.roll_dir, self._roll_dir_state = classify_axis(
            angle_deg=droll,
            prev_state=self._roll_dir_state,
            neg_label="ccw",
            zero_label="still",
            pos_label="cw",
            th_low=15.0,
            th_high=30.0,
        )

    def get_quaternion(self) -> Tuple[float, float, float, float]:
        """
        Called by IMUVisualizer to obtain the current quaternion
        (absolute orientation).
        """
        with self._lock:
            return self._w, self._x, self._y, self._z