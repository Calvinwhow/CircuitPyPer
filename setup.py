from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).resolve().parent


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
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.10,<3.14",
)
