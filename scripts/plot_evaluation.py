"""Export evaluation plots and optionally open the interactive viewer."""

import argparse
from pathlib import Path

from legged_runtime.evaluation.visualization import create_figures, export_figures, load_evaluation, show_viewer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Visualize a recorded LeggedRuntime evaluation.")
    parser.add_argument("--input", required=True, type=Path, help="Evaluation directory containing trajectory.npz.")
    parser.add_argument("--headless", action="store_true", help="Export PNGs without opening the GUI.")
    parser.add_argument("--no-save", action="store_true", help="Open GUI without automatically exporting PNGs.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG export resolution.")
    args = parser.parse_args()

    if args.headless and args.no_save:
        parser.error("--headless and --no-save cannot be combined")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    input_dir = ROOT / args.input
    data, metrics = load_evaluation(input_dir)
    figures = create_figures(data, metrics)

    if not args.no_save:
        output_dir = export_figures(figures, input_dir, dpi=args.dpi)
        print(f"[INFO] Plots saved to: {output_dir}")

    if not args.headless:
        try:
            show_viewer(input_dir, figures, metrics, dpi=args.dpi)
        except ImportError as exc:
            raise SystemExit(f"GUI unavailable: {exc}. Install Tkinter or use --headless.") from exc
        except Exception as exc:
            # Avoid hiding programming errors; only add guidance for Tk initialization failures.
            if type(exc).__name__ == "TclError":
                raise SystemExit(f"GUI unavailable: {exc}. Use --headless on a server without a display.") from exc
            raise


if __name__ == "__main__":
    main()
