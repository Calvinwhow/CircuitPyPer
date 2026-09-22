"""Circuit Viewer adapter for rendering fiber optimization frames."""

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image

try:
    from circuit_viewer.session import RenderSession
    from circuit_viewer.spec import new_fiber_layer, new_overlay, new_spec
except ModuleNotFoundError as exc:
    if exc.name != "circuit_viewer":
        raise
    viewer_root = Path(os.environ.get(
        "CIRCUIT_VIEWER_ROOT",
        Path(__file__).resolve().parents[4].parent.parent / "circuit_viewer",
    )).expanduser()
    if not (viewer_root / "circuit_viewer" / "__init__.py").is_file():
        raise ModuleNotFoundError(
            "circuit-viewer is required. Install it or set CIRCUIT_VIEWER_ROOT."
        ) from exc
    sys.path.insert(0, str(viewer_root))
    from circuit_viewer.session import RenderSession
    from circuit_viewer.spec import new_fiber_layer, new_overlay, new_spec


class CircuitFiberFrames:
    """Reuse one atlas, one render session, and one temporary value vector."""

    def __init__(self, atlas_path, fiber_io, *, vmax, max_lines=10000,
                 size=(900, 560), viewer_url=None):
        self.atlas_path = Path(atlas_path).expanduser().resolve()
        self.fiber_io = fiber_io
        self.size = size
        self.temporary = tempfile.TemporaryDirectory(prefix="optimization_fibers_")
        self.values_path = Path(self.temporary.name) / "frame.npy"
        np.save(self.values_path, np.zeros(len(fiber_io.reference_fibers),
                                           dtype=np.float32))

        overlay = new_overlay(
            nifti=str(self.values_path), name="Convergent map",
            palette="coolwarm", clim=[-float(vmax), float(vmax)],
            symmetric=True, threshold=0,
        )
        layer = new_fiber_layer(
            name="Iteration 0", source=str(self.atlas_path),
            coloring="value", color="#b7bcc4", overlays=[overlay],
            max_lines=max_lines, step=2, smooth=2,
        )
        self.scene = new_spec([layer])
        if viewer_url:
            with urlopen(viewer_url.rstrip("/") + "/api/spec", timeout=10) as response:
                template = json.load(response)
            for field in ("background", "camera", "center", "distance",
                          "lighting", "parallel"):
                if field in template:
                    self.scene[field] = template[field]
        self.session = RenderSession(self.scene)

    def render(self, values, iteration):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        mask = self.fiber_io.fiber_mask
        if values.size == len(self.fiber_io.reference_fibers):
            full_values = values
        elif mask is not None and values.size == int(mask.sum()):
            full_values = self.fiber_io.unmask_array(values, mask, fill_value=0)
        else:
            raise ValueError(
                f"Map has {values.size} fiber values, but the atlas has "
                f"{len(self.fiber_io.reference_fibers)} fibers."
            )
        np.save(self.values_path, np.asarray(full_values, dtype=np.float32))
        # A changed layer name refreshes its colors while cached geometry stays.
        self.scene["layers"][0]["name"] = f"Iteration {iteration + 1}"
        self.session.set_spec(self.scene)
        # macOS VTK creates a Cocoa window even off screen; Cocoa requires the
        # render call on the main thread. This CLI renders all frames there.
        render = self.session._render if sys.platform == "darwin" else self.session.render
        png = render(
            size=self.size, quality="preview", scalar_bar=False,
        )
        with Image.open(io.BytesIO(png)) as image:
            return np.asarray(image.convert("RGB"))

    def close(self):
        close = getattr(self.session, "close", None)
        if callable(close):
            close()
        else:
            try:
                import pyvista as pv
                pv.close_all()
            except Exception:
                pass
        self.temporary.cleanup()
