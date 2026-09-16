from pathlib import Path

from circuit_pyper.scripts.neuro_plotter import dispatch, FIGURES

colors = {
    0: "#c15656",  # Motor
    1: "#5071a0",  # Cognitive
    2: "#9a8ed1",  # Emotional
}

labels = {
    0: "motor",
    1: "cognitive",
    2: "emotional",
}

paths = {
    "fibers": (
        "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/"
        "symptom_on_lhs/fiber_regressions_clusters/cluster_regression_identity_standardized/"
        "Fiber_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional/regression/"
        "contrast_tval_FWE_{i}_positive_ftr.mat"
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

for modality, template in paths.items():
    for i in range(3):

        source = Path(template.format(i=i))

        output_dir = source.parent / "neuro_plots" / labels[i]

        dispatch(
            source_nifti=source,
            output_dir=output_dir,
            figures=FIGURES,
            overrides={
                "cmap": colors[i],
                "plot_kwargs": {
                    "color": colors[i],
                },
            },
            make_html=True,
            open_html=False,
        )