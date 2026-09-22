#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -eq 0 ]]; then
    echo "Do not run neuro_plotter_loop.sh with sudo; it needs the active Python environment." >&2
    echo "Run: scripts/neuro_plotter_loop.sh" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ -n ${VIRTUAL_ENV:-} && -x ${VIRTUAL_ENV}/bin/python ]]; then
    python_bin=${VIRTUAL_ENV}/bin/python
else
    python_bin=${PYTHON:-python3}
fi

exec "${python_bin}" "${script_dir}/neuro_plotter_loop.py" "$@"
