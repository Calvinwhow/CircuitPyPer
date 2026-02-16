from pathlib import Path
from setuptools import setup, find_packages
def parse_requirements(path):
    lines = Path(path).read_text().splitlines()
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
)