import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

def yield_files_recursively(root, pattern="*"):
    for file in Path(root).rglob(pattern):
        yield file

def compress_file(nii_path: Path) -> None:
    """Compress a .nii file to .nii.gz and delete the original."""
    gz_path = nii_path.with_suffix('.nii.gz')
    with nii_path.open('rb') as src, gzip.open(gz_path, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    nii_path.unlink()

def recursive_gzip(dir) -> None:
    dpath = Path(dir)
    for nii in tqdm(yield_files_recursively(dpath, '*.nii')):
        try:
            compress_file(nii)
        except Exception as e:
            print(f"Error compressing {nii}: {e}")
            continue

if __name__ == "__main__":
    dir = ''
    recursive_gzip(dir)