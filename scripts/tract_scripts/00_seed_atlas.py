import argparse
from calvin_utils.neuroimaging_utils.tract_utils.connectome_seed import ConnectomeSeed

### Script-style configurable args ###
CONNECTOME_PATH="/Volumes/HowExp/resources/connectome_dMRI/zarr5GB/zarr_vectors_test_BRIDGE_outputs_S-21-46_res-50um_space-MNI152_desc-bridgeanatomical.trk"
SEED_MASK="/Users/cu135/hires_backdrops/suit/atl-Anatom_space-MNI_dseg_coverage.nii.gz"
INTERSECT_MASK=["/Volumes/HowExp/resources/atlases/mni_space/MNI_structures/subcortex/subcortex_mask_2mm.nii.gz"]
EXCLUDE_MASK=[]
OUT="/Volumes/OneTouch/resources/zarr5GB"
MAX_FIBERS=200000

### CLI-style runner args ###
def argparser():
    parser = argparse.ArgumentParser(description="Seed a connectome with an atlas mask.")
    parser.add_argument("--connectome-path", default=CONNECTOME_PATH)
    parser.add_argument("--seed-mask", default=SEED_MASK)
    parser.add_argument("--intersect-mask", nargs="*", default=INTERSECT_MASK)
    parser.add_argument("--exclude-mask", nargs="*", default=EXCLUDE_MASK)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--max-fibers", type=int, default=MAX_FIBERS)
    return parser

### Public API ###
def main():
    args = argparser().parse_args()
    ConnectomeSeed(connectome_path=args.connectome_path, seed_mask=args.seed_mask, intersect_mask=args.intersect_mask, exclude_mask=args.exclude_mask, out=args.out, max_fibers=args.max_fibers)

if __name__ == "__main__":
    main()