# fiber_merger.py
import numpy as np


class FiberMerger:
    """
    Handles merging of every pair of completed selection groups, triggered by the 'm' key.

    Public API
    ----------
    FiberMerger(gui)
        gui : FiberSelectionGUI instance
    .on_m()               # triggered by 'm' key to merge groups
    .start_merge()        # external method to begin merge
    """

    def __init__(self, gui) -> None:
        self.gui = gui          # back-pointer to main GUI

    # ------------------------------------------------------------------
    # hot-key dispatcher
    def on_m(self):
        self._merge_all_groups()
    
    def start_merge(self):
        '''External reference to begin merge'''
        self._merge_all_groups()

    # ------------------------------------------------------------------
    def _merge_all_groups(self):
        groups = self.gui._sel_groups
        if len(groups) != 2:
            print("Need exactly 2 completed groups before merging. Press R to reset, then select two groups of fibers.")
            return

        # merge consecutive pairs: (0,1), (2,3), …
        while len(groups) >= 2:
            g1 = groups.pop(0)
            g2 = groups.pop(0)
            self._merge_two_groups(g1, g2)

        # clear highlights and reset state
        for idx in self.gui._highlighted[:]:
            self.gui._remove_highlight(idx)
        self.gui._selected.clear()
        self.gui._highlighted.clear()
        self.gui._sel_groups.clear()
        self.gui._compact_scene()
        self.gui._pl.render()
        print("Merge complete.")

    # ------------------------------------------------------------------
    def _merge_two_groups(self, g1_idx: list[int], g2_idx: list[int]) -> None:
        """
        Greedy pairing **between** two groups:
        • both fibres may be independently reversed so that
          the *last point of f1* meets the *first point of f2*
          at the smallest Euclidean distance.
        • unmatched fibres are simply discarded.
        """
        g1 = [(i, self.gui.streamlines[i]) for i in g1_idx]
        g2 = [(j, self.gui.streamlines[j]) for j in g2_idx]

        # build candidate list: (dist, i, j, rev1, rev2)
        cand = []
        for i, f1 in g1:
            for j, f2 in g2:
                # four orientation combos
                combos = [
                    (np.linalg.norm(f1[-1] - f2[0]),  False, False),
                    (np.linalg.norm(f1[-1] - f2[-1]), False, True ),
                    (np.linalg.norm(f1[ 0] - f2[0]),  True,  False),
                    (np.linalg.norm(f1[ 0] - f2[-1]), True,  True ),
                ]
                d, rev1, rev2 = min(combos, key=lambda x: x[0])
                cand.append((d, i, j, rev1, rev2))

        # greedy global assignment
        cand.sort(key=lambda x: x[0])
        used1, used2 = set(), set()

        for _, i, j, rev1, rev2 in cand:
            if i in used1 or j in used2:
                continue
            used1.add(i); used2.add(j)

            f1 = self.gui.streamlines[i][::-1] if rev1 else self.gui.streamlines[i]
            f2 = self.gui.streamlines[j][::-1] if rev2 else self.gui.streamlines[j]
            merged = np.vstack([f1, f2])

            self.gui._add_line(merged)               # visual feedback
            self.gui._pl.remove(self.gui._lines[i])  # drop originals
            self.gui._pl.remove(self.gui._lines[j])
            
            self.gui._drop_line(i)
            self.gui._drop_line(j)
            
        for idx in (set(g1_idx) - used1) | (set(g2_idx) - used2):
            self.gui._pl.remove(self.gui._lines[idx])
            self.gui._drop_line(idx)