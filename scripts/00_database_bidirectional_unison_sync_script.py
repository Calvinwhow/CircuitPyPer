#!/usr/bin/env python3
"""
Plain, edit-the-variables-at-the-top runner for bidirectional database sync.

This uses Unison because it tracks state between roots. That matters for
database-like folders where new files should propagate in both directions, real
deletions should remain deletions, and conflicting edits should be surfaced
instead of silently duplicating or overwriting data.

This is intentionally not a CLI. Change the values in the CONFIG section, then
run this file directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================


# Set this to the local folder you want synchronized from the machine running
# this script. On the laptop this might be the laptop EHD; at home this might be
# the home-lab EHD. Do not leave this as None.
LOCAL_DATABASE_ROOT = "/Volumes/YourEHD/database"

# Set this to the remote server path. The format is:
#   ssh://<user>@<host>//absolute/path/on/server
# If your SSH config already defines an alias, use:
#   ssh://<alias>//absolute/path/on/server
# Do not leave this as None.
REMOTE_DATABASE_ROOT = "ssh://your_server_alias//absolute/path/to/database"

# Usually leave as "unison". Change only if Unison is installed at a specific
# path, e.g. "/opt/homebrew/bin/unison".
UNISON_CMD = "unison"

# Set True to print the command and stop before launching Unison. This is not a
# Unison dry-run; it is a script-level preview. Set False to run Unison.
PREVIEW_COMMAND_ONLY = True

# Set True only when you want Unison to sync without interactive prompts. For
# database syncs, False is safer because conflicts should be reviewed manually.
BATCH = False

# Set True if you want non-conflicting changes accepted automatically while
# conflicts still stop for review. Keep False for the first run on new roots;
# set True later if the sync behavior is stable and you want less prompting.
AUTO_ACCEPT_NON_CONFLICTS = False

# Set True when you want Unison to detect updates by checking file contents
# instead of using faster filesystem metadata shortcuts. Slower, but safer for
# important data. This maps to Unison "-fastcheck false".
CHECK_CONTENTS = True

# Set True if you want Unison to preserve Unix permissions where possible.
# Leave False when syncing across systems/users where permissions differ.
SYNC_PERMISSIONS = False

# Set True if you want Unison to preserve modification times. Usually True is
# useful for analysis outputs and database snapshots.
SYNC_MODTIMES = True

# Leave as None to use Unison's default state directory. Set to a stable local
# path if you want this sync pair's archive/log state isolated. Do not casually
# change this after syncing, because Unison's archive state is what lets it
# distinguish real deletions from old files that should be resurrected.
# Example: "/Users/cu135/.unison_database_sync"
UNISON_STATE_DIR = None

# Set to a local folder for backups of overwritten/deleted files. Leave as None
# to disable Unison backups. Backups are useful while validating a new sync pair.
BACKUP_DIR = None

# Ignore patterns. Fill these with files that should never propagate. Leave an
# empty list if every file under the database root should be synchronized.
# Unison pattern examples:
#   "Name .DS_Store"
#   "Name __pycache__"
#   "Path tmp"
#   "Regex .*\\.log$"
IGNORE_PATTERNS = [
    "Name .DS_Store",
    "Name ._.*",
]

# Prefer patterns are dangerous for bidirectional database sync because they
# auto-resolve differences in favor of one side. Leave empty unless you have a
# specific class of generated files where one side should always win.
# Examples:
#   "newer"
#   LOCAL_DATABASE_ROOT
#   REMOTE_DATABASE_ROOT
PREFER_PATTERNS = []

# Add any project-specific Unison flags here. Leave empty unless needed.
# Examples:
#   "-follow", "Path symlinked_folder"
#   "-ignore", "Path cache"
EXTRA_UNISON_ARGS = []

# Optional logfile. Leave as None to use Unison's default log location.
LOG_FILE = None


# =============================================================================
# VALIDATION
# =============================================================================


def require_unison():
    """Fail early if Unison is not available."""
    if shutil.which(UNISON_CMD) is None and not Path(UNISON_CMD).exists():
        raise FileNotFoundError(
            f"Could not find Unison command: {UNISON_CMD}. "
            "Install Unison or set UNISON_CMD to its full path."
        )


def validate_roots():
    """Validate required sync roots."""
    if not LOCAL_DATABASE_ROOT:
        raise ValueError("LOCAL_DATABASE_ROOT must be set.")
    if not REMOTE_DATABASE_ROOT:
        raise ValueError("REMOTE_DATABASE_ROOT must be set.")

    local_root = Path(LOCAL_DATABASE_ROOT).expanduser()
    if not local_root.is_dir():
        raise FileNotFoundError(f"LOCAL_DATABASE_ROOT does not exist: {local_root}")


def validate_conflict_policy():
    """Prevent accidental silent conflict handling."""
    if PREFER_PATTERNS:
        print("PREFER_PATTERNS is set. Matching conflicts may be auto-resolved.")

    if BATCH and PREFER_PATTERNS:
        print("BATCH=True and PREFER_PATTERNS is set; review this carefully before running.")


# =============================================================================
# BUILD COMMAND
# =============================================================================


def add_flag(command, condition, flag):
    """Append a Unison flag when condition is true."""
    if condition:
        command.append(flag)


def build_unison_command():
    """Build the Unison command from the editable config above."""
    command = [
        UNISON_CMD,
        str(Path(LOCAL_DATABASE_ROOT).expanduser()),
        REMOTE_DATABASE_ROOT,
    ]

    command.extend(["-ui", "text"])
    add_flag(command, BATCH, "-batch")
    add_flag(command, AUTO_ACCEPT_NON_CONFLICTS, "-auto")
    add_flag(command, SYNC_MODTIMES, "-times")
    add_flag(command, True, "-sortnewfirst")

    if CHECK_CONTENTS:
        command.extend(["-fastcheck", "false"])

    if not SYNC_PERMISSIONS:
        command.extend(["-perms", "0"])

    if BACKUP_DIR:
        backup_dir = Path(BACKUP_DIR).expanduser()
        backup_dir.mkdir(parents=True, exist_ok=True)
        command.extend([
            "-backup",
            "Name *",
            "-backuploc",
            "central",
            "-backupdir",
            str(backup_dir),
        ])

    for pattern in IGNORE_PATTERNS:
        command.extend(["-ignore", pattern])

    for pattern in PREFER_PATTERNS:
        command.extend(["-prefer", pattern])

    if LOG_FILE:
        command.extend(["-logfile", str(Path(LOG_FILE).expanduser())])

    command.extend(EXTRA_UNISON_ARGS)
    return command


# =============================================================================
# RUN SYNC
# =============================================================================


def print_command(command):
    """Print a shell-readable command for inspection."""
    print("Running Unison command:")
    print(" ".join(shlex_quote(part) for part in command))


def shlex_quote(value):
    """Small local quote helper to keep imports minimal."""
    value = str(value)
    if not value:
        return "''"
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-")
    if all(char in safe_chars for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_bidirectional_sync():
    """Run one bidirectional sync between the local database root and server."""
    require_unison()
    validate_roots()
    validate_conflict_policy()

    command = build_unison_command()
    env = os.environ.copy()

    if UNISON_STATE_DIR:
        state_dir = Path(UNISON_STATE_DIR).expanduser()
        state_dir.mkdir(parents=True, exist_ok=True)
        env["UNISON"] = str(state_dir)

    print(f"Local root:  {Path(LOCAL_DATABASE_ROOT).expanduser()}")
    print(f"Remote root: {REMOTE_DATABASE_ROOT}")
    print(
        "PREVIEW_COMMAND_ONLY="
        f"{PREVIEW_COMMAND_ONLY}, BATCH={BATCH}, "
        f"AUTO_ACCEPT_NON_CONFLICTS={AUTO_ACCEPT_NON_CONFLICTS}"
    )
    if UNISON_STATE_DIR:
        print(f"Unison state dir: {env['UNISON']}")
    print_command(command)

    if PREVIEW_COMMAND_ONLY:
        print("Preview only. Set PREVIEW_COMMAND_ONLY = False to run Unison.")
        return

    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    run_bidirectional_sync()
