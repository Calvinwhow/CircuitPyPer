"""Executable wrapper for :mod:`calvin_utils.server_utils.job_submitter`."""

from runpy import run_module


if __name__ == "__main__":
    run_module("calvin_utils.server_utils.job_submitter", run_name="__main__")

