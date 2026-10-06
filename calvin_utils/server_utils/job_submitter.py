"""Build and optionally submit batches of LSF jobs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def build_batch_scripts(
    *,
    processes: int,
    batch_size: int,
    submission_commands: dict[str, str],
    output_dir: str | Path = ".",
) -> list[Path]:
    """Create one ``bsub`` script per process batch and return their paths."""
    if processes < 1 or batch_size < 1:
        raise ValueError("processes and batch_size must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, _ in enumerate(range(0, processes, batch_size)):
        commands = submission_commands.copy()
        if "-J" in commands:
            commands["-J"] = f"{commands['-J']}{index}"
        for stream in ("-o", "-e"):
            if stream in commands:
                base = Path(commands[stream])
                commands[stream] = str(base / f"{commands.get('-J', 'batch')}.txt")
        command = " ".join(f"{key} {value}" for key, value in commands.items())
        path = destination / f"batch_{index}.sh"
        path.write_text(f"#!/bin/bash\nbsub {command}\n", encoding="utf-8")
        paths.append(path)
    return paths


def submit_batch_scripts(paths: list[Path], *, dry_run: bool = False) -> None:
    """Submit prepared scripts, or print them in dry-run mode."""
    for path in paths:
        if dry_run:
            print(path.read_text(encoding="utf-8").rstrip())
        else:
            subprocess.run(["bash", str(path)], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script")
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--job-name", default="calvin_batch")
    parser.add_argument("--queue", default="normal")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    commands = {
        "-J": args.job_name,
        "-q": args.queue,
        "-w": str(args.batch_size),
        "python": str(Path(args.script).resolve()),
    }
    paths = build_batch_scripts(
        processes=args.processes,
        batch_size=args.batch_size,
        submission_commands=commands,
        output_dir=args.output_dir,
    )
    submit_batch_scripts(paths, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
