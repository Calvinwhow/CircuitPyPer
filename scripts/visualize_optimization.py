#!/usr/bin/env python3
"""Render a compact optimization history as a GIF and native map stack.

Example:
    python scripts/visualize_optimization.py results/optimization_history.npz \
        results/optimization.gif --max-frames 60 \
        --viewer-url http://127.0.0.1:8731
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.visualization import (
    render_optimization_history,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="Optimization history .npz")
    parser.add_argument(
        "output", type=Path, nargs="?",
        help="Output GIF path (default: optimization.gif beside the history)",
    )
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument(
        "--max-frames", type=int, default=60,
        help="Maximum evenly sampled GIF frames and saved map snapshots",
    )
    parser.add_argument("--view", choices=("auto", "spatial", "circuit", "metrics"), default="auto")
    parser.add_argument("--output-type", help="Override the history's image type")
    parser.add_argument("--mask-path", help="Override the history's image mask or fiber atlas")
    parser.add_argument("--vmax", type=float, help="Fixed absolute limit for spatial colors")
    parser.add_argument("--max-lines", type=int, default=10000,
                        help="Maximum fibers drawn per Circuit Viewer frame")
    parser.add_argument("--viewer-url",
                        help="Read camera and lighting from a running Circuit Viewer")
    parser.add_argument(
        "--map-stack", type=Path,
        help=("Output 4D NIfTI or 2D fiber NumPy path (default: use the GIF "
              "stem and the native extension)"),
    )
    parser.add_argument(
        "--no-map-stack", action="store_true",
        help="Render only the GIF and do not save reconstructed map snapshots",
    )
    args = parser.parse_args()
    path = render_optimization_history(
        args.history, args.output, fps=args.fps, max_frames=args.max_frames,
        view=args.view, output_type=args.output_type,
        mask_path=args.mask_path, vmax=args.vmax, max_lines=args.max_lines,
        viewer_url=args.viewer_url, map_stack_path=args.map_stack,
        write_map_stack=not args.no_map_stack,
    )
    print(f"Saved optimization GIF to: {path}")


if __name__ == "__main__":
    main()
