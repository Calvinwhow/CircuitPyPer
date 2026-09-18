#!/usr/bin/env python3
"""Render the configured regression result maps with neuro_plotter."""

import argparse
from pathlib import Path
import sys
import traceback


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from circuit_pyper.scripts.circuit_viewer_orchestrator import (  # noqa: E402
    FIGURES,
    dispatch,
    input_kind,
    write_collection_index,
)


COLORS = {
    0: "#c15656",  # Motor
    1: "#5071a0",  # Cognitive
    2: "#9a8ed1",  # Emotional
}

LABELS = {
    0: "motor",
    1: "cognitive",
    2: "emotional",
}

PATHS = {
    "fibers": (
        "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/"
        "symptom_on_lhs/fiber_regressions_clusters/cluster_regression_identity_standardized/"
        "Fiber_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional/regression/"
        "contrast_tval_FWE_{i}.fib.desc.json"
    ),
    "network": (
        "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/"
        "symptom_on_lhs/network_regressions_clusters/cluster_regression_identity_standardized/"
        "Nifti_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional/regression/"
        "contrast_tval_FWE_{i}.nii.gz"
    ),
    "vlsm": (
        "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/"
        "symptom_on_lhs/vlsm_regressions_clusters/cluster_regression_identity_standardized/"
        "Nifti_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional/regression/"
        "contrast_tval_FWE_{i}.nii.gz"
    ),
}


def _figure_roots(source):
    """Current and legacy plot roots, kept exactly where they already live."""
    current = source.parent / "neuro_plots"
    roots = [current]
    legacy = source.parent.parent / "neuro_plots"
    if legacy != current and legacy.is_dir():
        roots.append(legacy)
    return current, roots


def main(index_only=False):
    failures = []
    for modality, template in PATHS.items():
        sources = [Path(template.format(i=index)) for index in range(3)]
        for index in range(3):
            source = sources[index]
            output_dir = source.parent / "neuro_plots" / LABELS[index]

            if index_only:
                continue
            print(f"Rendering {modality}/{LABELS[index]}: {source}")
            try:
                tract_overrides = (
                    {
                        "tract_sign": "positive",
                    }
                    if modality == "fibers"
                    else {}
                )
                dispatch(
                    source_nifti=source,
                    output_dir=output_dir,
                    figures=FIGURES,
                    overrides={
                        "cmap": COLORS[index],
                        **tract_overrides,
                        "plot_kwargs": {
                            "color": COLORS[index],
                        },
                    },
                    make_html=True,
                    open_html=False,
                )
            except Exception as exc:
                failures.append((modality, LABELS[index], exc))
                traceback.print_exc()

        plot_root, figure_roots = _figure_roots(sources[0])
        volume_sources = {
            LABELS[index]: source
            for index, source in enumerate(sources)
            if source.is_file() and input_kind(source) == "nifti"
        }
        write_collection_index(
            output_dir=plot_root,
            title=f"{modality.title()} regression result maps",
            figure_roots=figure_roots,
            volume_niftis=volume_sources,
        )

    if failures:
        print(f"\nCompleted with {len(failures)} failed render(s):", file=sys.stderr)
        for modality, label, exc in failures:
            print(f"  {modality}/{label}: {exc}", file=sys.stderr)
        return 1
    return 0


def parser():
    result = argparse.ArgumentParser()
    result.add_argument(
        "--index-only",
        action="store_true",
        help="rebuild collection indexes from existing files without rendering",
    )
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(main(index_only=arguments.index_only))
