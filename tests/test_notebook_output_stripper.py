"""Behavioral tests for repository notebook-output hygiene."""

import copy
import json

from scripts.strip_notebook_outputs import clean_notebook, strip_notebook


def _notebook_with_output():
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["Keep this content."],
            },
            {
                "cell_type": "code",
                "execution_count": 7,
                "metadata": {"tags": ["keep-me"]},
                "outputs": [{"name": "stdout", "output_type": "stream", "text": ["large"]}],
                "source": ["print('large')"],
            },
        ],
        "metadata": {"kernelspec": {"name": "python3"}, "widgets": {"state": {}}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_clean_notebook_removes_only_generated_state():
    notebook = _notebook_with_output()
    expected_content = copy.deepcopy(notebook)

    assert clean_notebook(notebook)
    assert notebook["cells"][1]["outputs"] == []
    assert notebook["cells"][1]["execution_count"] is None
    assert "widgets" not in notebook["metadata"]
    assert notebook["cells"][0] == expected_content["cells"][0]
    assert notebook["cells"][1]["source"] == expected_content["cells"][1]["source"]
    assert notebook["cells"][1]["metadata"] == expected_content["cells"][1]["metadata"]
    assert notebook["metadata"]["kernelspec"] == expected_content["metadata"]["kernelspec"]


def test_check_mode_reports_without_writing_and_cleanup_is_idempotent(tmp_path):
    notebook_path = tmp_path / "example.ipynb"
    original_text = json.dumps(_notebook_with_output())
    notebook_path.write_text(original_text, encoding="utf-8")

    assert strip_notebook(notebook_path, check=True)
    assert notebook_path.read_text(encoding="utf-8") == original_text

    assert strip_notebook(notebook_path)
    assert not strip_notebook(notebook_path, check=True)


def test_empty_legacy_notebook_placeholder_is_ignored(tmp_path):
    notebook_path = tmp_path / "placeholder.ipynb"
    notebook_path.touch()

    assert not strip_notebook(notebook_path, check=True)
    assert notebook_path.read_bytes() == b""
