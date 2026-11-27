"""Offline trajectory visualization and optional interactive evaluation viewer."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def load_evaluation(input_dir):
    """Load an experiment without retaining an open NPZ file handle."""
    input_dir = Path(input_dir)
    with np.load(input_dir / "trajectory.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    metrics_path = input_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    return data, metrics


def create_figures(data, metrics=None):
    """Create plots supported by the available standardized trajectory fields.

    No robot names, joint counts, or policy observation indices are assumed.
    Actual states are sampled at time; commands were applied at observation_time.
    """
    metrics = metrics or {}
    time = np.asarray(data["time"], dtype=float)
    if time.ndim != 1 or not time.size or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Expected a nonempty, finite, strictly increasing time vector")

    observation_time = np.asarray(data.get("observation_time", time), dtype=float)
    if observation_time.shape != time.shape:
        raise ValueError("observation_time must match time")
    if not np.all(np.isfinite(observation_time)) or np.any(np.diff(observation_time) <= 0) or np.any(observation_time > time + 1e-9):
        raise ValueError("Invalid observation timestamps")

    start = observation_time[0]
    t = time - start
    t_cmd = observation_time - start
    warmup = float(metrics.get("warmup_s", 0.0))
    figures = {}

    def field(name, shape):
        if name not in data:
            return None
        value = np.asarray(data[name], dtype=float)
        if value.shape != (len(t), *shape):
            raise ValueError(f"Invalid shape for {name}: {value.shape}")
        return value

    def new_figure(name, title, rows=1):
        fig = Figure(figsize=(9, 2.5 * rows + 1.5), layout="constrained")
        axes = fig.subplots(rows, 1, sharex=True, squeeze=False)[:, 0]
        fig.suptitle(title, fontsize=14)
        figures[name] = fig
        return axes

    def format_time_axis(ax, ylabel, legend=False):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if warmup > 0:
            ax.axvspan(0, min(warmup, t[-1]), color="0.8", alpha=0.3, zorder=0)
        if legend:
            ax.legend(loc="best")

    def plot_command(ax, values):
        # Hold each command over its control interval, including the final interval.
        ax.step(np.r_[t_cmd, t[-1]], np.r_[values, values[-1]], where="post", linestyle="--", label="Commanded")

    actual = field("actual_velocity", (3,))
    command = field("velocity_command", (3,))
    labels = ("vx [m/s]", "vy [m/s]", "wz [rad/s]")

    if actual is not None:
        axes = new_figure("velocity_tracking", "Velocity tracking", 3)
        for i, ax in enumerate(axes):
            ax.plot(t, actual[:, i], label="Actual")
            if command is not None:
                plot_command(ax, command[:, i])
            format_time_axis(ax, labels[i], legend=True)
        axes[-1].set_xlabel("Elapsed simulation time [s]")

    if actual is not None and command is not None:
        axes = new_figure("velocity_error", "Velocity tracking error (actual - commanded)", 3)
        for i, ax in enumerate(axes):
            ax.plot(t, actual[:, i] - command[:, i])
            ax.axhline(0.0, linestyle="--", color="0.4", linewidth=0.8)
            format_time_axis(ax, labels[i])
        axes[-1].set_xlabel("Elapsed simulation time [s]")

    tilt = field("base_tilt", ())
    position = field("base_position", (3,))
    stability = []
    if tilt is not None:
        stability.append((tilt, "Base tilt [deg]", metrics.get("fall_detection", {}).get("max_base_tilt_deg")))
    if position is not None:
        stability.append((position[:, 2], "Base height [m]", metrics.get("fall_detection", {}).get("min_base_height_m")))
    if stability:
        axes = new_figure("base_stability", "Base stability", len(stability))
        for ax, (values, label, threshold) in zip(axes, stability):
            ax.plot(t, values, label="Actual")
            # Avoid flattening small variations when the fall threshold is far away.
            if threshold is not None:
                ax.text(0.99, 0.96, f"Fall threshold: {threshold:g}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
            format_time_axis(ax, label)
        axes[-1].set_xlabel("Elapsed simulation time [s]")

    height = field("torso_height", ())
    height_command = field("torso_height_command", ())
    if height is not None:
        axes = new_figure("torso_height_tracking", "Torso height tracking")
        axes[0].plot(t, height, label="Actual")
        if height_command is not None:
            plot_command(axes[0], height_command)
        format_time_axis(axes[0], "Height [m]", legend=True)
        axes[0].set_xlabel("Elapsed simulation time [s]")

    if position is not None:
        fig = Figure(figsize=(7, 6), layout="constrained")
        figures["xy_trajectory"] = fig
        ax = fig.subplots()
        fig.suptitle("Base trajectory in the world frame", fontsize=14)
        ax.plot(position[:, 0], position[:, 1], label="Trajectory")
        ax.scatter(*position[0, :2], marker="o", label="First sample", zorder=3)
        ax.scatter(*position[-1, :2], marker="x", label="Last sample", zorder=3)
        ax.set(xlabel="World x [m]", ylabel="World y [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    latency = field("inference_latency", ())
    if latency is not None:
        ax = new_figure("inference_latency", "Policy inference latency")[0]
        ax.plot(t_cmd, latency, label="Per inference", linewidth=0.9)
        ax.axhline(np.mean(latency), linestyle="--", label="Mean")
        ax.axhline(np.percentile(latency, 95), linestyle=":", label="95th percentile")
        format_time_axis(ax, "Latency [ms]", legend=True)
        ax.set_xlabel("Elapsed simulation time [s]")

    return figures


def export_figures(figures, input_dir, dpi=160):
    """Export to a new timestamped directory, never overwrite previous exports."""
    if not figures:
        raise ValueError("No supported trajectory plots to export")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(input_dir) / "plots" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    for name, fig in figures.items():
        # Save using an Agg canvas if the figure is not attached to a GUI yet.
        if not isinstance(fig.canvas, FigureCanvasAgg):
            FigureCanvasAgg(fig)
        fig.savefig(output_dir / f"{name}.png", dpi=dpi)
    return output_dir


def show_viewer(input_dir, figures, metrics, dpi=160):
    """Display interactive figures in one Tk window; imported only for GUI use."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title("LeggedRuntime | Evaluation Viewer")
    root.geometry("1200x800")
    root.minsize(800, 550)

    toolbar = ttk.Frame(root, padding=8)
    toolbar.pack(fill="x")
    experiment = tk.StringVar()
    status = tk.StringVar(value="Ready")
    ttk.Label(toolbar, textvariable=experiment).pack(side="left", fill="x", expand=True)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    ttk.Label(root, textvariable=status, padding=(8, 0, 8, 8)).pack(fill="x")

    current = {"path": Path(input_dir), "figures": figures, "metrics": metrics}
    canvases = []
    tab_names = {
        "velocity_tracking": "Velocity", "velocity_error": "Velocity errors",
        "base_stability": "Stability", "torso_height_tracking": "Torso height",
        "xy_trajectory": "XY trajectory", "inference_latency": "Latency",
    }

    def populate():
        for canvas in canvases:
            # Cancel deferred draws before destroying embedded Tk widgets.
            widget = canvas.get_tk_widget()
            for attribute in ("_idle_draw_id", "_event_loop_id"):
                callback = getattr(canvas, attribute, None)
                if callback is not None:
                    widget.after_cancel(callback)
                    setattr(canvas, attribute, None)
        canvases.clear()
        for tab_id in notebook.tabs():
            notebook.nametowidget(tab_id).destroy()

        experiment.set(f"Experiment: {current['path'].name}")
        summary = ttk.Frame(notebook, padding=10)
        notebook.add(summary, text="Summary")
        text = tk.Text(summary, wrap="none", font="TkFixedFont")
        scrollbar = ttk.Scrollbar(summary, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        text.insert("1.0", json.dumps(current["metrics"], indent=2) if current["metrics"] else "No metrics.json available.")
        text.configure(state="disabled")

        for name, fig in current["figures"].items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_names.get(name, name.replace("_", " ").title()))
            canvas = FigureCanvasTkAgg(fig, master=frame)
            nav = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            nav.update()
            nav.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            canvas.draw()
            canvases.append(canvas)

        if len(notebook.tabs()) > 1:
            notebook.select(1)
        status.set(str(current["path"]))

    def open_experiment():
        selected = filedialog.askdirectory(parent=root, initialdir=current["path"].parent, title="Select evaluation directory")
        if not selected:
            return
        try:
            data, new_metrics = load_evaluation(selected)
            new_figures = create_figures(data, new_metrics)
        except Exception as exc:
            messagebox.showerror("Could not load experiment", str(exc), parent=root)
            return
        current.update(path=Path(selected), figures=new_figures, metrics=new_metrics)
        populate()

    def export_current():
        try:
            path = export_figures(current["figures"], current["path"], dpi=dpi)
            status.set(f"Exported PNGs: {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=root)

    ttk.Button(toolbar, text="Open Experiment", command=open_experiment).pack(side="left", padx=4)
    ttk.Button(toolbar, text="Export PNGs", command=export_current).pack(side="left", padx=4)
    populate()
    root.mainloop()
