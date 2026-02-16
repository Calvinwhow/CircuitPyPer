# ensure Python 3.10.14 is available
#  ---RUN THESE FIRST---
# brew install python@3.10
#           OR
# sudo apt install python3.10

# Then run all of these
pyenv local 3.10.14
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .