#!/usr/bin/env python3
"""MRIcroGL snapshot rendering integration."""

import os
import subprocess
import imageio
from typing import List

class MRIcroGLSnapshotter:
    """
    A class that:
      1) Finds all NIfTI files in a given folder.
      2) Builds a Python-like MRIcroGL script to load each NIfTI and save a PNG snapshot.
      3) Executes MRIcroGL in batch mode to run that script.
      4) Reads the PNG frames and assembles them into a GIF.
    """

    def __init__(
        self,
        nifti_folder: str,
        mricrogl_exec: str,
        output_dir: str,
        script_name: str = "mricrogl_snapshot.py",
        gif_name: str = "snapshots.gif",
        fps: int = 8
    ):
        """
        :param nifti_folder: Folder containing existing NIfTI files.
        :param mricrogl_exec: Full path to the MRIcroGL executable.
        :param output_dir: Folder where the MRIcroGL script, PNGs, and final GIF will be placed.
        :param script_name: Name of the generated MRIcroGL script.
        :param gif_name: Name of the final GIF to be created.
        :param fps: Frames per second for the resulting GIF.
        """
        self.nifti_folder = os.path.abspath(nifti_folder)
        self.mricrogl_exec = os.path.abspath(mricrogl_exec)
        self.output_dir = os.path.abspath(output_dir)
        self.script_name = script_name
        self.gif_name = gif_name
        self.fps = fps

        self.png_dir = os.path.join(self.output_dir, "png_frames")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.png_dir, exist_ok=True)

        # Gather NIfTI files sorted by name
        self.nifti_files = self._get_nifti_files()
        if not self.nifti_files:
            raise FileNotFoundError(f"No NIfTI files found in {self.nifti_folder}.")

    def _get_nifti_files(self) -> List[str]:
        """
        Returns a sorted list of .nii or .nii.gz files in self.nifti_folder.
        Adjust the pattern to match your naming scheme if necessary.
        """
        all_files = os.listdir(self.nifti_folder)
        nifti_files = [
            f for f in all_files
            if f.lower().endswith(".nii") or f.lower().endswith(".nii.gz")
        ]
        nifti_files.sort()
        return nifti_files

    def _build_mricrogl_script(self, view='front', clip='none'):
        """
        Creates a Python-like script for MRIcroGL in batch mode.
        Overlays each NIfTI file twice (positive and negative),
        sets overlay surface style, sets viewpoint, optional clip,
        and saves a PNG for each map.
        
        :param view: One of ['front','back','left','right','top','bottom'].
        :param clip: Either 'none' or 'sagittal'.
        """
        import os
        
        # 1) Define standard viewpoints via azimuth/elevation
        view_dict = {
            'front':   (180, 0),
            'back':    (0, 0),
            'left':    (90, 0),
            'right':   (270, 0),
            'top':     (180, 90),
            'bottom':  (180, -90),
        }
        if view not in view_dict:
            raise ValueError(f"Invalid view '{view}'. Choose from {list(view_dict.keys())}")
        
        azim, elev = view_dict[view]

        # 2) Determine clip code
        #   This is just one example of a sagittal clip. Customize as needed.
        clip_lines = []
        if clip == 'sagittal':
            # For example, we clip half the brain from a sagittal plane
            # (0.4, 0, 160) is just an example from the docs
            clip_lines.append("gl.clipazimuthelevation(0.4, 0, 160)")
        elif clip != 'none':
            raise ValueError(f"Invalid clip mode '{clip}'. Choose 'none' or 'sagittal'.")

        script_path = os.path.join(self.output_dir, self.script_name)

        lines = []
        lines.append("import gl")
        lines.append("gl.resetdefaults()")
        lines.append("gl.backcolor(255,255,255)")  # White background

        # If you want the same background each time:
        lines.append("gl.loadimage('spm152')") 
        lines.append("overlaymaskedbybackground(True)") # Mask anything not on background

        lines.append(f"nifti_folder = r'{self.nifti_folder}'")
        lines.append(f"png_folder   = r'{self.png_dir}'")

        # Gather NIfTI filenames (already stored by this class)
        lines.append("nifti_files = [")
        for f in self.nifti_files:
            lines.append(f"    r'{f}',")
        lines.append("]")

        # 3) Build loop
        lines.append("for idx, nifti_name in enumerate(nifti_files):")
        lines.append("    # Close any previous overlays to start fresh each iteration")
        lines.append("    gl.overlaycloseall()")

        #   3a) Load the same file TWICE: once for positive, once for negative
        #       The 'background' is already loaded (spm152) by gl.loadimage() above
        lines.append("    nifti_path = f'{nifti_folder}/{nifti_name}'")

        #   Positive overlay
        lines.append("    pos_idx = gl.overlayload(nifti_path)")
        lines.append("    gl.overlayminmax(pos_idx, 0.00001, 0.30)")
        lines.append("    gl.colorname(pos_idx, '8redyell')")

        #   Negative overlay
        lines.append("    neg_idx = gl.overlayload(nifti_path)")
        lines.append("    gl.overlayminmax(neg_idx, -0.30, -0.000001)")
        lines.append("    gl.colorname(neg_idx, '6bluegreen')")

        #   3b) Switch to overlaySurface shader
        lines.append("    gl.shadername('overlaySurface')")

        #   3c) Optionally apply clipping
        if clip_lines:
            # e.g. gl.clipazimuthelevation(0.4, 0, 160)
            for cline in clip_lines:
                lines.append(f"    {cline}")

        #   3d) Choose viewpoint
        lines.append(f"    gl.azimuthelevation({azim}, {elev})")

        #   3e) Save a PNG
        lines.append("    out_png = f'{png_folder}/frame_{idx:03d}.png'")
        lines.append("    gl.savebmp(out_png)")

        # 4) Quit after finishing the loop
        lines.append("gl.quit()")

        # 5) Write out the script
        with open(script_path, "w") as f:
            f.write("\n".join(lines))


    def _run_mricrogl_script(self):
        """
        Runs MRIcroGL in 'headless' mode, passing our script with -s.
        """
        script_path = os.path.join(self.output_dir, self.script_name)
        cmd = [self.mricrogl_exec, "-s", script_path]
        subprocess.run(cmd, check=True)

    def _assemble_gif(self):
        """
        Reads all PNGs from self.png_dir and assembles them into a single GIF.
        """
        png_files = sorted([f for f in os.listdir(self.png_dir) if f.endswith(".png")])
        if not png_files:
            raise FileNotFoundError(f"No PNG frames found in {self.png_dir}.")

        frames = []
        for png in png_files:
            img_path = os.path.join(self.png_dir, png)
            frames.append(imageio.imread(img_path))

        out_gif_path = os.path.join(self.output_dir, self.gif_name)
        imageio.mimsave(out_gif_path, frames, fps=self.fps)
        print(f"GIF created at {out_gif_path}")
        
    def run_snapshot_pipeline(self):
        """
        Top-level method:
          1) Build the MRIcroGL script that loops over each NIfTI.
          2) Run that script in MRIcroGL (batch).
          3) Assemble all PNGs into a GIF.
        """
        self._build_mricrogl_script()
        self._run_mricrogl_script()
        self._assemble_gif()


# # --- Example usage ---
# if __name__ == "__main__":
#     # Suppose you have a folder of NIfTIs: "/path/to/my_niftis/"
#     # and an MRIcroGL executable at "/Applications/MRIcroGL.app/Contents/MacOS/MRIcroGL"
#     # plus an output directory where you want to store the PNGs and final GIF.

#     nifti_folder = "/path/to/my_niftis"
#     mricrogl_exec = "/Applications/MRIcroGL.app/Contents/MacOS/MRIcroGL"
#     output_dir = "./mricrogl_output"

#     snapshotter = MRIcroGLSnapshotter(
#         nifti_folder=nifti_folder,
#         mricrogl_exec=mricrogl_exec,
#         output_dir=output_dir,
#         script_name="mricrogl_snapshot.py",
#         gif_name="my_snaps.gif",
#         fps=5
#     )

#     # Run the pipeline:
#     #  - Build MRIcroGL script
#     #  - Run MRIcroGL to create PNGs
#     #  - Assemble PNGs -> GIF
#     snapshotter.run_snapshot_pipeline()
