#!/usr/bin/env python3
"""
End-to-end rerun: symptom regressions -> clustering -> cluster regressions ->
cerebellar flatmaps.

    caffeinate -i python circuit_pyper/scripts/run_overnight_pipeline.py

Each stage runs the same scripts you would run by hand, with their own CONFIG
sections; nothing here rewrites them. The tree names below must match the
OUT_DIR in those scripts, and the run stops if they don't -- a mismatch would
silently cluster the previous run's maps.

Every stage logs under LOG_DIR and records completion in STATE_FILE, so a
re-run skips finished stages. Inside stage 1 the pipelines already skip
analyses whose outputs exist.

Expect roughly 8 hours; stage 1 dominates (~165 s per symptom-modality at 1000
permutations).

This is not a CLI. Change the values in the CONFIG section, then run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CIRCUIT_PYPER_DIR = SCRIPT_DIR.parent
REPO_ROOT = CIRCUIT_PYPER_DIR.parent          # the pipelines import circuit_pyper.*


# =============================================================================
# CONFIG
# =============================================================================

SCA_ROOT = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs"

# The symptom-regression tree per modality. Must equal OUT_DIR in the stage 1
# script beside it; checked before anything runs.
TREES = {
    "network": "network_regressions_HigherIsWorse-RankedNaN",
    "vlsm": "vlsm_regressions_HigherIsWorse-RankedNaN",
    "fiber": "fiber_regressions_HigherIsWorse-RankedNaN",
}
SYMPTOM_PIPELINES = {
    "network": "regression_pipeline-lnm.py",
    "vlsm": "regression_pipeline-vlsm.py",
    "fiber": "regression_pipeline-fib.py",
}
CLUSTER_PIPELINES = {                   # stage 3, one per modality
    "network": "regression_pipeline-lnmClusterReg.py",
    "vlsm": "regression_pipeline-vlsmClusterReg.py",
    "fiber": "regression_pipeline-fibClusterReg.py",
}

# Stage 2. Clusters <tree>/ into <tree><CLUSTER_SUFFIX>/.
CLUSTER_SCRIPT = "cluster_pipeline.py"
CLUSTER_SUFFIX = "_Clusters"
CLUSTER_MODE = "both"           # "sweep" | "final" | "both"; the scores changed,
                                # so let it search rather than reuse old params
COLS_TO_FLIP = ""               # every score in the sheet is already higher-is-worse
FIBER_ATLAS = "/Volumes/OneTouch/resources/Atlas_tck_MNI/Atlas_all30_MNI.npz"

# Stage 4.
FIGURE_SCRIPT = "cerebellar_cluster_figures.py"

LOG_DIR = os.path.join(SCA_ROOT, "_overnight_logs")
STATE_FILE = os.path.join(LOG_DIR, "state.json")
STAGES = ["symptom_regressions", "clustering", "cluster_regressions", "figures"]
STOP_ON_STAGE_FAILURE = True    # each stage consumes the previous one's output
PYTHON = sys.executable


# =============================================================================
# PLUMBING
# =============================================================================


def say(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def load_state():
    path = Path(STATE_FILE)
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            say("state file unreadable; starting a fresh record")
    return {"done": [], "failed": [], "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}


def save_state(state):
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


def config_value(script, name):
    """The literal a script assigns to NAME at top level, or None."""
    text = (SCRIPT_DIR / script).read_text()
    match = re.search(rf"^{name}\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    return match.group(1) if match else None


def check_wiring():
    """Every script must point at the trees named above, or we stop."""
    problems = []
    for modality, tree in TREES.items():
        out_dir = config_value(SYMPTOM_PIPELINES[modality], "OUT_DIR")
        if out_dir != os.path.join(SCA_ROOT, tree):
            problems.append(f"{SYMPTOM_PIPELINES[modality]} OUT_DIR is {out_dir}, expected {tree}")
        expected_input = os.path.join(SCA_ROOT, tree + CLUSTER_SUFFIX, "cluster_regression_input.csv")
        got = config_value(CLUSTER_PIPELINES[modality], "INPUT_PATH")
        if got != expected_input:
            problems.append(f"{CLUSTER_PIPELINES[modality]} INPUT_PATH is {got}, expected {expected_input}")
    figures = (SCRIPT_DIR / FIGURE_SCRIPT).read_text()
    for tree in TREES.values():
        if tree + CLUSTER_SUFFIX not in figures:
            problems.append(f"{FIGURE_SCRIPT} has no cluster_dir for {tree + CLUSTER_SUFFIX}")
    if problems:
        for problem in problems:
            say(f"WIRING: {problem}")
        sys.exit("stopping: the scripts and this orchestrator disagree about where the results go")
    say("wiring checked: all four stages agree on the tree names")


def run(script, log_name, env_extra=None):
    """Run a script from the repo root, teeing its output to a log."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(CIRCUIT_PYPER_DIR), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update(env_extra or {})
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(LOG_DIR) / f"{log_name}.log"
    say(f"  {script} -> {log_path.name}")
    with log_path.open("a") as log:
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S}  {script} =====\n")
        log.flush()
        result = subprocess.run([PYTHON, str(SCRIPT_DIR / script)], cwd=str(REPO_ROOT),
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        say(f"  FAILED ({result.returncode}): see {log_path}")
    return result.returncode == 0


# =============================================================================
# STAGES
# =============================================================================


def stage_symptom_regressions():
    return all([run(SYMPTOM_PIPELINES[m], f"1_symptom_{m}") for m in TREES])


def stage_clustering():
    return run(CLUSTER_SCRIPT, "2_clustering", {
        "SCA_ROOT": SCA_ROOT,
        "ONLY_MODALITIES": ",".join(TREES),
        "SCA_TREE_NETWORK": TREES["network"],
        "SCA_TREE_VLSM": TREES["vlsm"],
        "SCA_TREE_FIBER": TREES["fiber"],
        "CLUSTER_SUFFIX": CLUSTER_SUFFIX,
        "CLUSTER_MODE": CLUSTER_MODE,
        "COLS_TO_FLIP": COLS_TO_FLIP,
        "FIBER_ATLAS": FIBER_ATLAS,
    })


def stage_cluster_regressions():
    return all([run(CLUSTER_PIPELINES[m], f"3_clusterreg_{m}") for m in TREES])


def stage_figures():
    return run(FIGURE_SCRIPT, "4_figures")


RUNNERS = {
    "symptom_regressions": stage_symptom_regressions,
    "clustering": stage_clustering,
    "cluster_regressions": stage_cluster_regressions,
    "figures": stage_figures,
}


def main():
    check_wiring()
    state = load_state()
    for stage in STAGES:
        if stage in state["done"]:
            say(f"{stage}: already done, skipping")
            continue
        say(f"{stage}: starting")
        if RUNNERS[stage]():
            state["done"].append(stage)
            state["failed"] = [s for s in state["failed"] if s != stage]
            save_state(state)
            say(f"{stage}: done")
        else:
            if stage not in state["failed"]:
                state["failed"].append(stage)
            save_state(state)
            say(f"{stage}: failed")
            if STOP_ON_STAGE_FAILURE:
                sys.exit(1)
    say(f"all stages complete; logs in {LOG_DIR}")


if __name__ == "__main__":
    main()
