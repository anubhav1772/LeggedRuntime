"""Offline MuJoCo trajectory replay with synchronized diagnostics."""

from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class MujocoReplay:
    def __init__(self, model_path, data, width, height, base_name, distance=2.8, azimuth=135.0, elevation=-15.0):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.qpos = np.asarray(data["qpos"])
        self.qvel = np.asarray(data["qvel"])
        n = len(data["time"])

        if self.qpos.shape != (n, self.model.nq) or self.qvel.shape != (n, self.model.nv):
            raise ValueError("Recorded state dimensions do not match the replay model")
        if not np.isfinite(self.qpos).all() or not np.isfinite(self.qvel).all():
            raise ValueError("Recorded states contain non-finite values")

        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_name)
        if self.base_id < 1:
            raise ValueError(f"Replay base body not found: {base_name}")

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = distance
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation

        # MuJoCo's offscreen buffer must be at least as large as the render target.
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

    def render(self, index, sim_time):
        self.data.qpos[:] = self.qpos[index]
        self.data.qvel[:] = self.qvel[index]
        self.data.time = sim_time
        self.mj.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.xpos[self.base_id]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def replay_timeline(data, fps):
    """Select recorded states on a uniform video clock, without interpolating quaternions."""
    time = np.asarray(data["time"], dtype=float)
    obs_time = np.asarray(data.get("observation_time", time), dtype=float)
    if time.ndim != 1 or len(time) == 0 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Expected nonempty, strictly increasing state timestamps")
    if obs_time.shape != time.shape or not np.isfinite(obs_time).all() or np.any(np.diff(obs_time) <= 0) or np.any(obs_time > time):
        raise ValueError("Invalid observation timestamps")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive")

    start = obs_time[0]
    elapsed = time - start
    # The initial state is not recorded: start at the first post-step state.
    video_time = elapsed[0] + np.arange(max(1, int(np.floor((elapsed[-1] - elapsed[0]) * fps + 1e-9)) + 1)) / fps
    indices = np.searchsorted(elapsed, video_time + 1e-9, side="right") - 1
    return elapsed, obs_time - start, np.clip(indices, 0, len(time) - 1)


class DiagnosticFrame:
    """One reusable Agg canvas containing simulation and recorded diagnostics."""

    def __init__(self, data, cfg, metrics, elapsed, command_time, width, height):
        self.data = data
        self.elapsed = elapsed
        self.command_time = command_time
        self.lines = []
        self.cursors = []
        self.fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        self.canvas = FigureCanvasAgg(self.fig)
        panels = []
        n = len(elapsed)

        def field(name, shape):
            if name not in data:
                return None
            value = np.asarray(data[name], dtype=float)
            if value.shape != (n, *shape) or not np.isfinite(value).all():
                raise ValueError(f"Invalid diagnostic field: {name}")
            return value

        actual = field("actual_velocity", (3,))
        command = field("velocity_command", (3,))
        if actual is not None:
            for i, label in enumerate(("Forward velocity [m/s]", "Lateral velocity [m/s]", "Yaw velocity [rad/s]")):
                panels.append((label, actual[:, i], command[:, i] if command is not None else None))

        torso = field("torso_height", ())
        tilt = field("base_tilt", ())
        if torso is not None:
            panels.append(("Torso height [m]", torso, field("torso_height_command", ())))
        elif tilt is not None:
            panels.append(("Base tilt [deg]", tilt, None))

        grid = self.fig.add_gridspec(max(1, len(panels)), 2, width_ratios=(1.2, 1), left=0.035, right=0.965, top=0.87, bottom=0.17, wspace=0.24, hspace=0.65)
        self.image_ax = self.fig.add_subplot(grid[:, 0])
        self.image_ax.set_axis_off()
        self.image = self.image_ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
        self.fig.text(0.035, 0.945, "LeggedRuntime  |  Simulation + runtime diagnostics", fontsize=19, weight="bold")
        self.fig.text(0.035, 0.91, f'{cfg.get("robot", "Robot")}  /  {cfg.get("policy", {}).get("type", "policy")}  /  Recorded trajectory replay', fontsize=11, color="0.35")
        self.status = self.fig.text(0.035, 0.10, "", fontsize=11)
        self.fig.text(0.035, 0.055, "Recorded states and measurements  •  Not a live physics rollout", fontsize=10, color="0.4")

        if not panels:
            ax = self.fig.add_subplot(grid[:, 1])
            ax.text(0.5, 0.5, "No supported diagnostics recorded", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        for i, (label, values, target) in enumerate(panels):
            ax = self.fig.add_subplot(grid[i, 1])
            # Use full-trajectory limits, so the axes do not jump between frames.
            bounds = values if target is None else np.concatenate([values, target])
            lo, hi = float(bounds.min()), float(bounds.max())
            padding = max(0.12 * (hi - lo), 0.015 * max(abs(lo), abs(hi), 1.0))
            ax.set_ylim(lo - padding, hi + padding)
            ax.set_xlim(0, max(elapsed[-1], 1e-3))
            ax.set_title(label, loc="left", fontsize=10, pad=5)
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=9, labelbottom=i == len(panels) - 1)
            if i == len(panels) - 1:
                ax.set_xlabel("Simulation time [s]", fontsize=10)
            line, = ax.plot([], [], linewidth=1.7, label="Actual")
            cmd_line = None
            if target is not None:
                cmd_line, = ax.step([], [], where="post", linestyle="--", linewidth=1.4, label="Commanded")
                ax.legend(loc="upper right", fontsize=8)
            self.lines.append((line, values, cmd_line, target))
            self.cursors.append(ax.axvline(elapsed[0], color="0.35", linewidth=0.8, alpha=0.7))

        self.tilt = tilt
        scale = width / 1920
        for text in self.fig.findobj(match=lambda obj: hasattr(obj, "get_fontsize") and hasattr(obj, "set_fontsize")):
            text.set_fontsize(text.get_fontsize() * scale)
        self.latency = field("inference_latency", ())
        self.fall_info = metrics.get("fall_detection") or {}
        self.fall_time = self.fall_info.get("time_to_fall_s")
        sim = cfg.get("simulation", {})
        dt, decimation = sim.get("dt"), sim.get("decimation")
        self.frequencies = f"Physics: {1 / dt:g} Hz   |   Policy: {1 / (dt * decimation):g} Hz" if dt and decimation else ""

    def render(self, image, index):
        self.image.set_data(image)
        # imshow's original extent comes from a 2x2 placeholder; update it for real frames.
        h, w = image.shape[:2]
        self.image.set_extent((-0.5, w - 0.5, h - 0.5, -0.5))
        self.image_ax.set_xlim(-0.5, w - 0.5)
        self.image_ax.set_ylim(h - 0.5, -0.5)
        t = self.elapsed[index]
        for line, values, cmd_line, target in self.lines:
            line.set_data(self.elapsed[:index + 1], values[:index + 1])
            if cmd_line is not None:
                cmd_line.set_data(np.r_[self.command_time[:index + 1], t], np.r_[target[:index + 1], target[index]])
        for cursor in self.cursors:
            cursor.set_xdata([t, t])

        status = [f"Time: {t:.2f} s"]
        if self.frequencies:
            status.append(self.frequencies)
        if self.tilt is not None:
            status.append(f"Tilt: {self.tilt[index]:.2f} deg")
        if self.latency is not None:
            status.append(f"Inference: {self.latency[index]:.2f} ms")
        if self.fall_time is not None:
            status.append(f"Fall: {'detected' if self.fall_time <= t else 'not yet detected'}")
        elif self.fall_info.get("fall_detected") is False:
            status.append("Fall: none detected")
        self.status.set_text("   |   ".join(status))
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()


def export_replay(data, cfg, metrics, input_dir, model_path, width=1920, height=1080, fps=30,
                  distance=2.8, azimuth=135.0, elevation=-15.0):
    import imageio.v2 as imageio

    if width < 640 or height < 360 or width % 2 or height % 2:
        raise ValueError("Video dimensions must be even and at least 640x360")
    if not np.isfinite([distance, azimuth, elevation]).all() or distance <= 0:
        raise ValueError("Invalid camera parameters")
    elapsed, command_time, indices = replay_timeline(data, fps)
    base_name = cfg.get("bodies", cfg.get("evaluation", {}).get("bodies", {})).get("base")
    if not base_name:
        raise ValueError("Configure bodies.base in the saved experiment configuration")

    frame = DiagnosticFrame(data, cfg, metrics, elapsed, command_time, width, height)
    replay = MujocoReplay(model_path, data, width // 2, int(height * 0.75), base_name, distance, azimuth, elevation)
    try:
        output_dir = Path(input_dir) / "videos" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=False)
        # Stream frames to FFmpeg; avoid keeping the entire video in memory.
        with imageio.get_writer(str(output_dir / "evaluation.mp4"), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2, quality=8, output_params=["-movflags", "+faststart"]) as writer:
            for k, index in enumerate(indices):
                rgb = replay.render(index, float(data["time"][index]))
                composite = frame.render(rgb, index)
                writer.append_data(composite)
                if k == len(indices) // 2:
                    imageio.imwrite(output_dir / "thumbnail.png", composite)
                if k % max(1, int(fps * 5)) == 0:
                    print(f"[INFO] Rendering frame {k + 1}/{len(indices)}")
    finally:
        replay.close()

    return output_dir
