"""
fibstitch.py
============

Interactive streamline viewer + selector based on **vedo**.
Click any fiber once → it turns **red** and gains a yellow glow.
Click again → it returns to its original colour.

If exactly two fibers are selected an optional *merge-handler* is invoked:
    merged = merge_handler.merge(fiber_a, fiber_b)
and the returned fiber is appended to the scene (lime-green).

API summary
-----------
FiberSelectionGUI(streamlines, *, merge_handler=None, window_title="…")
    .show()                 # blocking start of window
    .selected_indices       # list[int] – indices currently selected
    .clear_selection()      # programmatic deselect-all

Internal structure (readable sections, each with a docstring):

    _load()                 # I/O helper
    _build_scene()          # plotting, LUT
    _on_pick()              # callback dispatcher
    _select()/_deselect()   # selection bookkeeping
    _add_highlight()/_remove_highlight()
"""

from __future__ import annotations

import io
import gzip
import math
import numpy as np
from pathlib import Path
from typing import Iterable
from scipy.io import loadmat
from nibabel.streamlines import load as _load_trk

from pathlib import Path
from typing import Iterable
from vedo import Line, Plotter
from vedo import Point as VPoint, Line as VLine         # avoid name clash
from vedo.colors import build_lut, color_map
from calvin_utils.tract_utils.fibsel import FiberSelector
from calvin_utils.tract_utils.fibmerge import FiberMerger
from calvin_utils.tract_utils.fibsave import FibSave

class FiberStich:
    """
    Lightweight streamline visualiser with click-to-select.

    Parameters
    ----------
    streamlines : path | (iterable of (N,3) arrays)
        Provide a `.trk`/`.tck` filename 
        **or**
        Any iterable whose elements are (N, 3) float arrays.
    window_title : str
        Title of the render window.

    Public attributes
    -----------------
    selected_indices : list[int]
        Indices of currently selected fibers (read-only).
    """

    # ──────────────────────────────────────────────────────────────────── init
    def __init__(self, streamlines, window_title="Fiber-Selection GUI"):
        """
        Initialize the Fiber-Selection GUI.
        Parameters
        ----------
        streamlines : iterable of np.ndarray
            The collection of streamlines (fibers) to be visualized and manipulated.
        window_title : str, optional
            The title of the GUI window. Default is "Fiber-Selection GUI".
        Attributes
        ----------
        streamlines : list of np.ndarray
            Loaded full-resolution fibers.
        _n : int
            Number of streamlines.
        display_streams : list of np.ndarray
            Decimated streamlines for display purposes.
        _visible : np.ndarray of bool
            Boolean mask indicating which streamlines are visible.
        _lines : list
            List of line objects in the plotter.
        _orig_cols : list
            Original colors of the streamlines.
        _selected : list
            Indices of selected streamlines.
        _highlighted : list
            Indices of highlighted streamlines.
        _sel_groups : list
            Groups of selected streamlines.
        _pl : Plotter
            Vedo Plotter instance for visualization.
        _selector : FiberSelector
            Helper for fiber selection.
        _merger : FiberMerger
            Helper for merging fibers.
        Notes
        -----
        This method loads the streamlines, initializes visibility, sets up the
        visualization scene, and adds interaction callbacks.
        """

        # ---- load full-resolution fibres
        self.streamlines: list[np.ndarray] = list(self._load(streamlines))
        self._n = len(self.streamlines)
        # ---- decimate the fibers to ease visualization
        self.display_streams = [self._decimate(f, 8) for f in self.streamlines]
        step = self._get_steps()
        self._visible = np.zeros(self._n, bool)
        self._visible[::step] = True

        self._lines, self._orig_cols = [], []
        self._selected, self._highlighted, self._sel_groups = [], [], []

        # ---- Vedo plotter & helpers
        self._pl = Plotter(title=window_title, axes=1, bg="black", interactive=False)
        self._selector = FiberSelector(self)
        self._merger   = FiberMerger(self)
        self._saver    = FibSave(self, streamlines)

        # ---- draw and callbacks
        self._build_scene()
        self._add_callbacks()
        self._print_instructions()

    ### Public ###
    @property
    def selected_indices(self) -> list[int]:
        """Return indices of all currently selected fibers."""
        return self._selected

    def show(self) -> None:
        """Blocking start of the interactive vedo wisndow."""
        self._pl.show(interactive=True)
    
    ### Helpers ###
    def _print_instructions(self):
        print("\nFiberSelectionGUI Instructions:")
        print("  - Left click: Select/deselect a fiber (turns red/yellow when selected).")
        print("  - Right click: Deselect all fibers.")
        print("  - 'r' key: Reset GUI and clear selections.")
        print("  - 's' key: Start fiber selection gate.")
        print("  - 'm' key: Merge currently selected fibers (if exactly two are selected).")
        print("  - 'w' key: Save current selection/scene.")
        print("\nNote: Fibers are downsampled for visualization speed, but all operations (e.g., merging, saving) use the full-resolution data.")
        
    def _get_steps(self):
        '''target max of 1000 fibers on screen at any time for processing purposes.'''
        return max(1, self._n // 1000)
    
    @staticmethod
    def _decimate(arr: np.ndarray, step: int = 8):
        '''Downsample fiber for visualization'''
        if len(arr) <= 2 * step:
            return arr
        keep = np.r_[0,
                    np.arange(step, len(arr) - step, step),
                    len(arr) - 1]
        return arr[keep]

    ### Callback Methods ###
    def _add_callbacks(self) -> None: 
        print("Pick mode activated. Click to select a fiber.")
        print("Right click to de-select fibers.")
        self.lb_cb_id = self._pl.add_callback("LeftButtonPress", self._on_p)
        self._pl.add_callback("RightButtonPress", lambda evt: (self._clear_selection(), self._pl.render()))
        self._pl.add_callback("KeyPress", self._on_key)
    
    def _on_key(self, evt):
        key = getattr(evt, "keypress",
                    getattr(evt, "keyPressed", "")).lower()
        if key == "r":
            self._reset_gui()
        elif key == "s":
            self._selector.start_gate()        # new public helper, see below
        elif key == "m":
            self._merger.start_merge()                # merges current groups
        elif key == "w":
            self._saver.save()
        
    def _reset_gui(self):
        self._clear_selection()           # remove highlights & groups
        # remove any gate leftovers if helper is mid-session
        if hasattr(self._selector, "_gate_actors"):
            for a in self._selector._gate_actors:
                self._pl.remove(a)
            self._selector._gate_actors.clear()
            self._selector._gate_pts.clear()
            self._pl.clear()           # wipe every actor
            self._lines.clear()
            self._orig_cols.clear()
            self._build_scene()        # redraw original fibres

        print("GUI reset.")
        self._pl.render()
    
    def _on_p(self, evt) -> None:
        """Toggle selection and handle optional merging."""
        actor = evt.actor
        if actor is None or not hasattr(actor, "name"):
            return
        if not str(actor.name).isdigit():
            return
        idx = int(actor.name)
        if idx in self._selected:
            self._deselect(idx)
        else:
            self._select(idx)
        self._pl.render()
        
    def _clear_selection(self) -> None:
        """Programmatically deselect everything."""
        for i in self._highlighted[:]:
            self._remove_highlight(i)
        self._selected.clear()
        self._highlighted.clear()
        self._sel_groups.clear()

    ### Fiber Methods ###
    @staticmethod
    def _load(src) -> Iterable[np.ndarray]:
        """Yield (N,3) float arrays from path or iterable."""
        if isinstance(src, (list, tuple, np.ndarray)):
            for arr in src:
                a = np.asarray(arr, float)
                if a.ndim != 2 or a.shape[1] != 3:
                    raise ValueError("Each streamline must be shaped (N, 3).")
                yield a
            return
        src = Path(src)
        ext = src.suffix.lower()
        if src.name.endswith('.tt.gz') or ext == '.tt': 
            yield from FiberStich._iter_tt(src)
        else:
            yield from _load_trk(src).streamlines
            
    @staticmethod
    def _iter_tt(src: Path) -> Iterable[np.ndarray]:
        """Yield (N,3) float32 streamlines from a DSI-Studio TinyTrack file."""
        if src.name.endswith(".tt.gz"):
            with gzip.open(src, "rb") as f:
                buf = f.read()
            mat = loadmat(io.BytesIO(buf), squeeze_me=True, struct_as_record=False)
        else:                                                # plain .tt
            mat = loadmat(src, squeeze_me=True, struct_as_record=False)

        tracks = mat.get("track")
        if tracks is None:
            raise ValueError("No variable named 'track' found in TT file")

        # tracks is an object array: each element is an (N,3) numeric array
        for tr in tracks:
            arr = np.asarray(tr, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError("Each streamline must be (N,3)")
            yield arr / 32.0          # DSI-Studio stores vertices in 1/32-voxel

    ### Plotting Methods ###
    def _build_scene(self):
        """Render a coloured actor only for fibres flagged in `_visible`."""
        colours = [color_map(i, "plasma", 0, self._n - 1) for i in range(self._n)]
        lut = build_lut([(i, c) for i, c in enumerate(colours)],
                        vmin=0, vmax=self._n - 1)

        for i, (fib, show) in enumerate(zip(self.display_streams, self._visible)):
            if show:
                rgb = lut.GetTableValue(i)[:3]
                ln = Line(fib, c=rgb, lw=2).lighting(False)
                ln.name = str(i); ln.pickable(True)
                self._pl += ln
            else:
                ln = None                              # placeholder

            self._lines.append(ln)
            self._orig_cols.append(rgb)

    def _add_highlight(self, idx: int):
        # lazily create a line if it was not rendered
        if self._lines[idx] is None:
            ln = Line(self.display_streams[idx], c="wheat", lw=1).lighting(False)
            ln.name = str(idx); ln.pickable(True)
            self._pl += ln
            self._lines[idx] = ln
        ln = self._lines[idx]
        ln.color("yellow").lw(3)
        if idx not in self._highlighted:
            self._highlighted.append(idx)

    def _remove_highlight(self, idx: int):
        hl = getattr(self._lines[idx], "_hl", None)
        if hl is not None:
            self._pl.remove(hl)
            del self._lines[idx]._hl
        if idx in self._highlighted:
            self._highlighted.remove(idx)

    def _add_line(self, fib: np.ndarray):
        self.streamlines.append(fib)
        disp = self._decimate(fib, 8)
        self.display_streams.append(disp)
        self._visible = np.append(self._visible, True)     # always show merged

        ln = Line(disp, c="lime", lw=2).lighting(False)
        ln.name = str(len(self._lines))
        ln.pickable(True)
        self._lines.append(ln)
        self._pl += ln


    def _drop_line(self, idx: int) -> None:
        """Remove actor idx and blank its slots (kept until we compact)."""
        act = self._lines[idx]
        if act is None:
            return
        self._pl.remove(act)
        self._lines[idx]      = None
        self.streamlines[idx] = None
        if idx in self._highlighted:
            self._highlighted.remove(idx)
        if idx in self._selected:
            self._selected.remove(idx)
            
    def _compact_scene(self) -> None:
        """
        Pack out all Nones from _lines / streamlines, renumber .name fields,
        and rebuild the index lists.
        """
        new_lines, new_streams = [], []
        remap = {}                           # old->new index
        for old_idx, (ln, st) in enumerate(zip(self._lines, self.streamlines)):
            if ln is None:
                continue
            new_idx = len(new_lines)
            remap[old_idx] = new_idx
            ln.name = str(new_idx)           # keep picking functional
            new_lines.append(ln)
            new_streams.append(st)

        self._lines       = new_lines
        self.streamlines  = new_streams
        # rebuild index-based lists
        self._highlighted = [remap[i] for i in self._highlighted]
        self._selected    = [remap[i] for i in self._selected]

    ### Selection Methods ###
    def _select(self, idx: int) -> None:
        """Mark one fiber as selected; update visuals."""
        self._selected.append(idx)
        self._add_highlight(idx)

    def _deselect(self, idx: int) -> None:
        """Undo selection visuals for given fiber."""
        self._selected.remove(idx)
        self._remove_highlight(idx)


if __name__ == "__main__":
    import numpy as np
    # Example: simulate three random fibers
    fibers = [np.cumsum(np.random.randn(n, 3), 0) for n in (20, 22, 24, 26, 28)]
    # fibers ='/Volumes/Expansion/atlases/tracts/circuit_of_papez/circuit_of_papez_BL.trk.gz'
    gui = FiberStich(
        streamlines=fibers,
        window_title="Fiber-Selection GUI"
    )
    gui.show()