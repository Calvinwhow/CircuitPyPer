"""
fib_save.py
===========

Tiny helper that dumps the *current* set of streamlines from a running
``FiberSelectionGUI`` to a `.trk` file that is fully NiBabel-compatible.

Usage
-----

```python
from fib_save import FibSave

saver = FibSave(gui, orig_trk_path)  # orig header cloned automatically
saver.save("/tmp/my_edited_bundle.trk")


### Key points

* **No decimation**
  - the class writes `gui.streamlines`, which always stores
  the un-altered, full-resolution data, plus any merged fibres you created.
* **Header cloning**  
  - if you loaded your original `.trk`, pass that path and
  the affine / voxel sizes are preserved; otherwise an identity affine is
  used.
* **Single dependency** 
  - requires *nibabel* ≥ 3.  
  (`pip install nibabel` if you haven't already.)
* **One-liner integration**
  - instantiate once, then call `.save()` whenever
  you want to dump the current state; wire it to a hot-key if you like.

"""
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence

class FibSave:
    """
    Parameters
    ----------
    gui
    The live FiberSelectionGUI instance - provides .streamlines.
    orig_trk : path-like | None
    If you loaded from a .trk file pass that same path so we can
    copy its affine & header. If None we fall back to identity affine
    and a minimal header that most viewers accept.
    """
    # ------------------------------------------------------------------
    def __init__(self, gui, orig_trk: str | Path | None = None) -> None:
        self._gui = gui
        self._affine, self._hdr = np.eye(4), None
        self._fname = None
        if isinstance(orig_trk, (str, os.PathLike, Path)) and Path(orig_trk).is_file():
            orig_trk = Path(orig_trk)
            t = nib.streamlines.load(str(orig_trk))
            self._affine = t.affine
            self._hdr    = t.header.copy()        # keep voxel-to-ras etc.

            self._dir = orig_trk.parent
            self._orig_fname = orig_trk.stem
            self._fname = f"{self._orig_fname}_fibstitched{orig_trk.suffix}"

    # ------------------------------------------------------------------
    def save(self) -> None:
        """
        Write *all* surviving fibres (merged + untouched) to ``out_path``.
        If out_path is not provided, use self._fname if available.
        """
        if self._fname is None:
            print("Error: Please provide a valid .trk file path or initialize with a proper .trk file.")
            return 
        out_name = os.path.join(self._dir, self._fname)
        streams = ArraySequence(self._gui.streamlines)   # full-res data
        tractogram = nib.streamlines.Tractogram(streams, affine_to_rasmm=self._affine)
        nib.streamlines.save(tractogram, out_name, header=self._hdr)
        print(f"Saved {len(streams)} streamlines → {out_name}")