from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent


def shared_data_files():
    """Install non-Python assets in a stable, platform-independent location."""
    groups = {}
    for source_root, destination_root in (
        (PROJECT_ROOT / "resources", Path("share/calvin_utils/resources")),
        (PROJECT_ROOT / "calvin_utils/shell_scripts", Path("share/calvin_utils/shell_scripts")),
    ):
        for source in source_root.rglob("*"):
            if not source.is_file() or source.name == ".DS_Store":
                continue
            destination = destination_root / source.relative_to(source_root).parent
            groups.setdefault(destination.as_posix(), []).append(
                source.relative_to(PROJECT_ROOT).as_posix()
            )
    return sorted(groups.items())


def parse_requirements(filename):
    lines = (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r") or line.startswith("--"):
            continue  # skip nested includes / pip flags
        reqs.append(line)
    return reqs

setup(
    name="calvin_utils",
    version="1.1.0",
    packages=find_packages(),
    package_data={"calvin_utils.testing_utils": ["nouns.txt"]},
    data_files=shared_data_files(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "ml": ["imbalanced-learn", "pysr", "scikit-optimize"],
        "tract": ["vedo"],
        "legacy": ["nltools", "pathlib2"],
    },
    python_requires=">=3.10,<3.14",
)
