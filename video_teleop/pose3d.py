"""Live COCO-17 lifting and an off-screen, mouse-rotatable 3D viewer."""

from collections import deque

import cv2
import numpy as np
import torch

from video_teleop.third_party.videopose3d.model import TemporalModelOptimized1f
from video_teleop.pose_visibility import COCO_NAMES, H36M_NAMES, coco_visibility, h36m_visibility
EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7),
    (7, 8), (8, 9), (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)


class PoseLifter:
    """The symmetric 243-frame model runs with repeated future frames for live use.

    Output: Human3.6M joints, estimated metres, camera axes (right, down,
    forward), pelvis at zero. History is never shared between people.
    """

    model_name = "VideoPose3D"
    temporal_mode = "live_repeated_future"

    def __init__(self, checkpoint, device, joint_names, threshold=0.5):
        names_to_index = {name: index for index, name in joint_names.items()}
        self.indices = [names_to_index[name] for name in COCO_NAMES]
        self.device = device
        self.threshold = threshold
        self.model = TemporalModelOptimized1f(17, 2, 17, [3, 3, 3, 3, 3])
        # This checkpoint is obtained from the official Facebook Research URL.
        # It contains NumPy RNG/optimizer state as well as tensor weights.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_pos"], strict=True)
        self.model.to(device).eval()
        self.half_window = (self.model.receptive_field() - 1) // 2
        self.history = deque(maxlen=self.half_window + 1)
        self.reset()

    def reset(self):
        self.history.clear()
        self.person_id = None
        self.previous_xy = None
        self.previous_xyz = None
        self.frame_size = None
        self.previous_valid = None
        self.observed = np.zeros(17, dtype=bool)

    @torch.inference_mode()
    def update(self, points, person_id, width, height):
        if person_id != self.person_id or self.frame_size != (width, height):
            self.reset()
            self.person_id = person_id
            self.frame_size = (width, height)
        if points is None:
            self.reset()
            return None
        points = points[self.indices]
        valid = coco_visibility(points, width, height, self.threshold)
        if valid.sum() < 3:
            self.reset()
            return None
        if self.previous_valid is not None and not np.array_equal(valid, self.previous_valid):
            # Do not smooth a newly visible part against its occluded predictions.
            self.history.clear()
            self.previous_xyz = None
        self.previous_valid = valid.copy()
        self.observed = h36m_visibility(valid)
        xy = points[:, :2][:, ::-1].copy()
        # The fixed-size network needs finite inputs. Placeholders remain internal;
        # their predictions are masked out before display, export and retargeting.
        xy[~valid] = xy[valid].mean(axis=0)
        self.previous_xy = xy.copy()
        xy = (xy / width * 2 - [1, height / width]).astype(np.float32)
        self.history.append(xy)
        window = np.pad(
            np.stack(self.history),
            ((self.half_window + 1 - len(self.history), self.half_window), (0, 0), (0, 0)),
            mode="edge",
        )
        prediction = self.model(torch.from_numpy(window[None]).to(self.device))[0, 0]
        xyz = prediction.cpu().numpy().copy()
        xyz -= xyz[0].copy()
        if not np.isfinite(xyz).all():
            self.reset()
            return None
        if self.previous_xyz is not None:
            xyz = 0.4 * xyz + 0.6 * self.previous_xyz
        self.previous_xyz = xyz.copy()
        xyz[~self.observed] = np.nan
        return xyz


class Pose3DViewer:
    """Render with Agg so both GUI and headless recording use the same view."""

    def __init__(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 5), dpi=100, facecolor="#161616")
        self.canvas = FigureCanvasAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_facecolor("#161616")
        self.figure.subplots_adjust(left=0.02, right=0.94, bottom=0.08, top=0.88)
        self.axes.set_xlabel("X: right (m)", color="white", labelpad=0)
        self.axes.set_ylabel("Y: depth (m)", color="white", labelpad=0)
        self.axes.set_zlabel("Z: up (m)", color="white", labelpad=0)
        self.axes.tick_params(colors="#aaaaaa", labelsize=7, pad=0)
        for axis in (self.axes.xaxis, self.axes.yaxis, self.axes.zaxis):
            axis.set_pane_color((0.13, 0.13, 0.13, 1))
        self.axes.set_box_aspect((1, 1, 1))
        self.lines = []
        for _, child in EDGES:
            color = "#56b4ff" if child in (4, 5, 6, 11, 12, 13) else "#ffad55"
            line, = self.axes.plot([], [], [], "o-", color=color, linewidth=2.5, markersize=4)
            self.lines.append(line)
        self.palm_lines = {side: self.axes.plot([], [], [], "o-", color="#40dfff", linewidth=3, markersize=5)[0]
                           for side in ("left", "right")}
        self.azimuth, self.elevation, self.radius = -65.0, 15.0, 1.0
        self.drag = None

    def mouse(self, event, x, y, flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            old_x, old_y = self.drag
            self.azimuth += (x - old_x) * 0.5
            self.elevation = float(np.clip(self.elevation + (y - old_y) * 0.5, -85, 85))
            self.drag = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.radius = float(np.clip(self.radius * (0.9 if flags > 0 else 1.1), 0.4, 3.0))

    def render(self, xyz, person_id, palms=None):
        # Camera (right, down, forward) -> displayed (right, depth, up).
        points = None if xyz is None else xyz[:, [0, 2, 1]] * [1, 1, -1]
        for line, (a, b) in zip(self.lines, EDGES):
            if points is None or not np.isfinite(points[[a, b]]).all():
                line.set_data_3d([], [], [])
            else:
                line.set_data_3d(*points[[a, b]].T)
        for side, wrist in (("left", 13), ("right", 16)):
            line = self.palm_lines[side]
            if xyz is None or not palms or side not in palms or not np.isfinite(xyz[wrist]).all():
                line.set_data_3d([], [], [])
            else:
                hand = np.stack((xyz[wrist], xyz[wrist] + palms[side].offset_camera))
                line.set_data_3d(*(hand[:, [0, 2, 1]] * [1, 1, -1]).T)
        r = self.radius
        self.axes.set(xlim=(-r, r), ylim=(-r, r), zlim=(-r, r))
        self.axes.view_init(elev=self.elevation, azim=self.azimuth)
        title = "No reliable pose" if xyz is None else f"Person {person_id} | estimated 3D pose"
        self.axes.set_title(title, color="white", fontsize=11)
        self.canvas.draw()
        return cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
