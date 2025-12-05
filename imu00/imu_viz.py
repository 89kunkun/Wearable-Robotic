from typing import Callable, Tuple
import numpy as np
from vedo import Plotter, Line, Text2D, settings


def quat_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-6:
        return np.eye(3)
    q /= norm
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])
    return R


class IMUVisualizer:
    def __init__(
        self,
        get_quaternion: Callable[[], Tuple[float, float, float, float]],
        dt_ms: int = 20,
    ):
        self.get_quaternion = get_quaternion
        self.dt_ms = dt_ms

        settings.use_parallel_projection = True

        self.plotter = Plotter(axes=1)
        self.text = Text2D("IMU orientation (AR/VR stabilized)", pos="top-left")

        origin = np.array([0.0, 0.0, 0.0])
        self.line_x = Line(origin, [1, 0, 0]).c("red4").lw(4)
        self.line_y = Line(origin, [0, 1, 0]).c("green4").lw(4)
        self.line_z = Line(origin, [0, 0, 1]).c("blue4").lw(4)

        self.plotter.show(
            self.line_x,
            self.line_y,
            self.line_z,
            self.text,
            viewup="z",
            resetcam=True,
            interactive=False,
        )

    def _timer_callback(self, evt):
        w, x, y, z = self.get_quaternion()
        R = quat_to_rotmat(w, x, y, z)

        origin = [0.0, 0.0, 0.0]
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        self.line_x.points = [origin, x_axis]
        self.line_y.points = [origin, y_axis]
        self.line_z.points = [origin, z_axis]

        self.plotter.render()

    def start(self):
        self.plotter.add_callback("timer", self._timer_callback)
        self.plotter.timer_callback("start", dt=self.dt_ms)

        self.plotter.show(interactive=True)

