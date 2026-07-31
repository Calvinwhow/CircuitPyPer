from calvin_utils.neuroimaging_utils.tract_utils.connectome_seed import ConnectomeSeed

seeded = ConnectomeSeed(
    connectome_path="/Volumes/HowExp/resources/connectome_dMRI/i74(Full)/data.trk",
    seed_mask="/Users/cu135/hires_backdrops/suit/atl-Anatom_space-MNI_dseg_coverage.nii.gz",
    intersect_mask=["/Volumes/HowExp/resources/atlases/mni_space/MNI_structures/subcortex/subcortex_mask_2mm.nii.gz"],
    exclude_mask=[],
    out="/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/connectomic/um1_fibers",
    max_fibers=200000
)
