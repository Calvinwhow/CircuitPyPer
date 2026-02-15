import os
from pathlib import Path
from typing import List

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


class NiftiAdder:
    """
    Add two or more co-registered NIfTIs voxel-wise and save the result.
    """

    # ------------------------------------------------------------------ #
    # constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(self, nifti_paths):
        if isinstance(nifti_paths, str):
            nifti_paths = [nifti_paths]
        if len(nifti_paths) < 2:
            raise ValueError("Need at least two NIfTIs to add.")
        self.nifti_paths = nifti_paths
        self.pairs: List[List[str]] = []

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _load_all(self):
        """Load all volumes as nibabel images."""
        return [nib.load(p) for p in self.nifti_paths]

    def _resample_to_first(self, imgs):
        """Resample images to the first image’s grid and sum."""
        ref = imgs[0]
        data = np.zeros(ref.shape, dtype=np.float32)
        for img in imgs:
            if img.shape != ref.shape or not np.allclose(img.affine, ref.affine):
                img = resample_from_to(img, (ref.shape, ref.affine), order=0)
            data += img.get_fdata(dtype=np.float32)
        return data, ref.affine, ref.header
    
    def _shared_substring(self):
        """
        Find the longest shared substring between the first two nifti_paths (ignoring extensions).
        """
        def strip_ext(p):
            return Path(p).stem.replace(".nii", "")
        s1 = strip_ext(self.nifti_paths[0])
        s2 = strip_ext(self.nifti_paths[1])
        # Find longest common substring
        m = [[0]*(1+len(s2)) for _ in range(1+len(s1))]
        longest, x_longest = 0, 0
        for x in range(1, 1+len(s1)):
            for y in range(1, 1+len(s2)):
                if s1[x-1] == s2[y-1]:
                    m[x][y] = m[x-1][y-1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                        x_longest = x
                else:
                    m[x][y] = 0
        return s1[x_longest-longest:x_longest] if longest > 0 else ""

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def find_pairs(directory: str, stem_length=None) -> List[List[str]]:
        """
        Return [[orig, derived], …] where one file-name is a substring of the
        other (extension ignored).
        """
        paths = sorted(Path(directory).iterdir())
        stems = [
            p.name[:-stem_length] if stem_length is not None
            else p.name[:-7] if p.name.endswith(".nii.gz")
            else p.name[:-4] if p.name.endswith(".nii")
            else p.name
            for p in paths
        ]

        used, pairs = set(), []
        for i, p_i in enumerate(paths):
            if i in used:
                continue
            for j in range(i + 1, len(paths)):
                if j in used:
                    continue
                if stems[i] in stems[j] or stems[j] in stems[i]:
                    pairs.append([str(p_i), str(paths[j])])
                    used.update({i, j})
                    break
        return pairs
    
    @staticmethod
    def add_and_save_pairs(pairs: list, out_dir: str, suffix: str = "_added") -> None:
        """
        Loop through `pairs` and write summed volumes for each pair.
        """
        for p1, p2 in pairs:
            adder = NiftiAdder([p1, p2])
            print(f"Saved: {adder.add_and_save(out_dir, suffix)}")

    def add_and_save(self, out_dir: str, suffix: str = "_added") -> str:
        """
        Sum `self.nifti_paths` and write one file named <first><suffix>.nii.gz.
        """
        os.makedirs(out_dir, exist_ok=True)
        summed, affine, hdr = self._resample_to_first(self._load_all())
        substring = self._shared_substring() 
        out_path = Path(out_dir) / f"{substring}{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(summed, affine, hdr), out_path)
        return str(out_path)