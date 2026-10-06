import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from calvin_utils.plotting_utils.simple_heatmap import simple_heatmap


def test_simple_heatmap_promotes_one_text_column_to_row_labels(tmp_path):
    data = pd.DataFrame(
        {
            "Symptom": ["Gait", "Limb Ataxia"],
            "Local": ["0.10", "-0.20"],
            "Network": [0.30, 0.40],
            "Fiber": [0.05, 0.15],
        }
    )

    fig, ax = plt.subplots()
    simple_heatmap(
        data,
        ax=ax,
        palette="RdBu",
        cbar_range=(-0.5, 0.5),
        out_dir=str(tmp_path),
    )

    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "Local",
        "Network",
        "Fiber",
    ]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == [
        "Gait",
        "Limb Ataxia",
    ]
    assert (tmp_path / "heatmap.svg").is_file()
    plt.close(fig)


def test_simple_heatmap_rejects_ambiguous_text_columns():
    data = pd.DataFrame(
        {
            "Symptom": ["Gait", "Limb Ataxia"],
            "Group": ["Motor", "Motor"],
            "Weight": [0.10, 0.20],
        }
    )

    with pytest.raises(ValueError, match="Heatmap values must be numeric"):
        simple_heatmap(data)

    plt.close("all")
