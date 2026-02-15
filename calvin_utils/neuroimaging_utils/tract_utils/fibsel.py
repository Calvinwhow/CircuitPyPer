# ────────────────────────── helpers.py ──────────────────────────
import numpy as np
import vtk
from vedo import Point as VPoint, Line as VLine


class FiberSelector:
    """
    FiberSelector handles interactive fiber selection within a GUI, typically for tractography visualization.
    This class enables users to select fibers by defining a gate (a plane) through two mouse clicks in the GUI.
    It manages the selection process, visual feedback, and group management for selected fibers.
    Parameters
    gui : FiberSelectionGUI
        The host GUI instance that provides rendering and selection context.
    Attributes
    gui : FiberSelectionGUI
        Reference to the main GUI instance.
    _gate_pts : list of np.ndarray
        List of points defining the gate (usually two points).
    _gate_actors : list
        Visual actors (markers and lines) used to display the gate in the GUI.
    _cb_id : int or None
        Callback ID for the gate mode mouse event handler.
    Methods
    -------
    on_s()
        Entry point for starting gate mode via keyboard shortcut.
    start_gate()
        Public method to initiate gate mode programmatically.
    _enter_gate_mode()
        Internal method to set up gate mode and register mouse callbacks.
    _exit_gate_mode()
        Internal method to clean up gate mode and remove visual actors.
    _on_click(evt)
        Internal callback for handling mouse clicks and collecting gate points.
    _finish_gate()
        Finalizes the gate, applies selection logic, and updates selection groups.
    _merge_groups()
        Merges two selection groups and clears their highlights.
    _toggle_by_plane(p0, p1)
        Toggles selection of fibers based on their position relative to the defined gate plane.
    Usage
    -----
    Typically, users interact with the FiberSelector via the GUI, clicking two points to define a selection gate.
    The class manages the selection logic and updates the GUI accordingly.
    """
    def __init__(self, gui) -> None:
        self.gui = gui                # back-pointer to main GUI
        self._gate_pts: list[np.ndarray] = []
        self._gate_actors: list = []   # white markers + helper line
        self._cb_id = None             # callback id for gate clicks

    # -----------------------------------------------------------------
    # external entry points
    def on_s(self):
        self._enter_gate_mode()
    
    def start_gate(self):
        """Public entry so GUI can start gate mode without touching events."""
        self._enter_gate_mode()

    # -----------------------------------------------------------------
    # internal helpers
    def _enter_gate_mode(self):
        print("Selection mode: click two points …")
        self._gate_pts.clear()
        pl = self.gui._pl
        self._cb_id = pl.add_callback("LeftButtonPress", self._on_click)

    def _exit_gate_mode(self):
        pl = self.gui._pl
        pl.remove_callback(self._cb_id)

        for a in self._gate_actors:
            pl.remove(a)
        self._gate_actors.clear()
        pl.render()

    def _on_click(self, evt):
        # display → world on focal plane
        try:
            x, y = evt.picked2d
        except AttributeError:
            x, y = evt.interactor.GetEventPosition()
        c = vtk.vtkCoordinate(); c.SetCoordinateSystemToDisplay()
        c.SetValue(float(x), float(y), 0)
        p = np.asarray(c.GetComputedWorldValue(self.gui._pl.renderer), float)

        self._gate_pts.append(p)
        m = VPoint(p, r=8, c="white")
        self.gui._pl += m; self._gate_actors.append(m)

        if len(self._gate_pts) == 2:
            self._finish_gate()

    def _finish_gate(self):
        p0, p1 = self._gate_pts
        seg = VLine([p0, p1], lw=3, c="white")
        self.gui._pl += seg; self._gate_actors.append(seg)

        self._toggle_by_plane(p0, p1)

        self.gui._sel_groups.append(self.gui._selected.copy())
        self.gui._selected.clear()

        self._exit_gate_mode()
        print("Gate selection complete.")

    # -----------------------------------------------------------------
    def _merge_groups(self):
        gA_idx, gB_idx = self.gui._sel_groups
        gA = [self.gui.streamlines[i] for i in gA_idx]
        gB = [self.gui.streamlines[i] for i in gB_idx]
        # clear both logical + visual selections
        for i in gA_idx + gB_idx:
            self.gui._remove_highlight(i)
        self.gui._sel_groups.clear()

    def _toggle_by_plane(self, p0, p1):
        cam = self.gui._pl.renderer.GetActiveCamera()
        n = np.cross(p1 - p0,
                     np.asarray(cam.GetDirectionOfProjection(), float))
        n /= (np.linalg.norm(n) + 1e-12)
        for idx, fib in enumerate(self.gui.streamlines):
            d = (fib.astype(float) - p0) @ n
            if np.any(d > 0) and np.any(d < 0):
                if idx in self.gui._selected:
                    self.gui._deselect(idx)
                else:
                    self.gui._select(idx)
