import json
import os
from typing import Iterable, List, Literal

import numpy as np
import pandas as pd

from calvin_utils.plotting_utils.simple_dynamite_plot import SimpleDynamitePlot, SimpleDynamitePlotPair
from calvin_utils.plotting_utils.simple_box_plot import SimpleBoxPlot, SimpleBoxPlotPair
from calvin_utils.plotting_utils.simple_histogram_plot import SimpleHistogramPlot
from calvin_utils.plotting_utils.simple_cdf_plot import SimpleCdfPlot, SimpleCdfPlotPair
from calvin_utils.plotting_utils.simple_line_plot import SimpleLinePlot
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot


class EvalStimParams:
    """
    Evaluate stimulation amplitudes stored as JSON arrays and generate single-group
    plots across contacts and across per-electrode sums.

    Each JSON is expected to contain a list like:
    [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    """

    def __init__(
        self,
        csv_path: str,
        *,
        json_col: str,
        out_dir: str | None = None,
        subject_col: str | None = None,
        plot_type: Literal["box", "dynamite", "hist", "cdf", "line"] = "box",
        contact_labels: Iterable[str] | None = None,
        contact_plot_title: str | None = 'Distribution of Current Across Contacts',
        contact_x_label: str | None = 'Contact',
        contact_y_label: str | None = 'Amplitude (mA)',
        sum_plot_title: str | None = 'Overall Amplitude',
        sum_x_label: str | None = '',
        sum_y_label: str | None = 'Amplitude (mA)',
    ):
        self.csv_path = csv_path
        self.json_col = json_col
        self.out_dir = out_dir
        self.subject_col = subject_col
        self.plot_type = plot_type
        self.contact_labels = list(contact_labels) if contact_labels is not None else None
        self.contact_plot_title = contact_plot_title
        self.contact_x_label = contact_x_label
        self.contact_y_label = contact_y_label
        self.sum_plot_title = sum_plot_title
        self.sum_x_label = sum_x_label
        self.sum_y_label = sum_y_label

        self._df = None
        self._arrays = []
        self._n_contacts = None

    # ---- setters/getters ----
    @property
    def csv_path(self):
        return self._csv_path

    @csv_path.setter
    def csv_path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("csv_path must be a non-empty string")
        self._csv_path = value

    @property
    def json_col(self):
        return self._json_col

    @json_col.setter
    def json_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("json_col must be a non-empty string")
        self._json_col = value

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, value):
        if value is None:
            self._out_dir = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("out_dir must be a non-empty string or None")
        self._out_dir = value

    @property
    def subject_col(self):
        return self._subject_col

    @subject_col.setter
    def subject_col(self, value):
        if value is None:
            self._subject_col = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("subject_col must be a non-empty string or None")
        self._subject_col = value

    # ---- public API ----
    def run(self):
        self._load_csv()
        self._validate_columns()
        self._load_json_arrays()
        self._append_overall_amplitude()
        contact_df = self._build_contact_dataframe()
        sum_df = self._build_sum_dataframe()
        self._plot_both(contact_df, sum_df)
        return contact_df, sum_df

    # ---- internal helpers ----
    def _load_csv(self):
        self._df = pd.read_csv(self.csv_path)

    def _validate_columns(self):
        if self.json_col not in self._df.columns:
            raise ValueError(f"Missing required column: {self.json_col}")
        if self.subject_col and self.subject_col not in self._df.columns:
            raise ValueError(f"subject_col '{self.subject_col}' not found in CSV")

    def _load_json_arrays(self):
        arrays = []
        keep_rows = []
        for idx, row in self._df.iterrows():
            p = row[self.json_col]
            if not isinstance(p, str) or not p.strip():
                print("Missing JSON path. Dropping row.")
                continue
            try:
                arr = self._read_json_list(p)
            except Exception as e:
                print(f"Invalid JSON path or format. Dropping row. Error: {e}")
                continue
            arrays.append(arr)
            keep_rows.append(idx)
        if not arrays:
            raise ValueError("No valid JSON arrays found.")
        if keep_rows:
            self._df = self._df.loc[keep_rows].reset_index(drop=True)
        self._validate_lengths(arrays)
        self._arrays = arrays

    def _read_json_list(self, path: str) -> List[float]:
        path = os.path.expanduser(path)
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON does not contain a list: {path}")
        return [float(x) for x in data]

    def _validate_lengths(self, arrays: List[List[float]]):
        lengths = {len(a) for a in arrays}
        if len(lengths) != 1:
            raise ValueError(
                "different electrode configurations detected, please restrict analyses to similar electrodes"
            )
        self._n_contacts = lengths.pop()
        if self.contact_labels is not None and len(self.contact_labels) != self._n_contacts:
            raise ValueError(
                f"contact_labels length ({len(self.contact_labels)}) "
                f"does not match contacts ({self._n_contacts})"
            )

    def _append_overall_amplitude(self):
        if self._arrays is None or not self._arrays:
            return
        sums = [float(np.sum(a)) for a in self._arrays]
        colname = f"{self.json_col}_amplitude"
        self._df[colname] = sums
        self._df.to_csv(self.csv_path, index=False)

    def _build_contact_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, arr in enumerate(self._arrays):
            for c_idx, val in enumerate(arr):
                label = self._contact_label(c_idx)
                rows.append({"electrode": i, "contact": label, "amplitude": val})
        return pd.DataFrame(rows)

    def _build_sum_dataframe(self) -> pd.DataFrame:
        sums = [float(np.sum(a)) for a in self._arrays]
        return pd.DataFrame({"electrode": list(range(len(sums))), "sum_amplitude": sums})

    def _contact_label(self, idx: int) -> str:
        if self.contact_labels is not None:
            return self.contact_labels[idx]
        return f"C{idx}"

    def _plot_contacts(self, contact_df: pd.DataFrame):
        if self.plot_type == "hist":
            return self._plot_contacts_hist(contact_df)
        if self.plot_type == "cdf":
            return self._plot_contacts_cdf(contact_df)
        if self.plot_type == "line":
            return self._plot_contacts_line(contact_df)
        title = self.contact_plot_title or "Contact amplitudes"
        plotter_cls = self._plotter_class()
        plotter = plotter_cls(
            contact_df,
            group_col="contact",
            value_col="amplitude",
            dataset_name=title,
            out_dir=None,
            xlabel=self.contact_x_label or "Contact",
            ylabel=self.contact_y_label or "Amplitude (mA)",
            order=self._contact_order(),
        )
        return plotter

    def _plot_sums(self, sum_df: pd.DataFrame):
        if self.plot_type == "hist":
            return self._plot_sums_hist(sum_df)
        if self.plot_type == "cdf":
            return self._plot_sums_cdf(sum_df)
        if self.plot_type == "line":
            return self._plot_sums_line(sum_df)
        plot_df = sum_df.assign(metric="Electrode sum")
        title = self.sum_plot_title or "Summed amplitude"
        plotter_cls = self._plotter_class()
        plotter = plotter_cls(
            plot_df,
            group_col="metric",
            value_col="sum_amplitude",
            dataset_name=title,
            out_dir=None,
            xlabel=self.sum_x_label or "",
            ylabel=self.sum_y_label or "Amplitude (mA)",
            order=["Electrode sum"],
        )
        return plotter

    def _contact_order(self):
        if self.contact_labels is not None:
            return list(self.contact_labels)
        return [f"C{i}" for i in range(self._n_contacts)]

    def _resolve_out_dir(self):
        if self.out_dir:
            return self.out_dir
        return os.path.dirname(os.path.abspath(self.csv_path))

    def _plotter_class(self):
        if self.plot_type == "box":
            return SimpleBoxPlot
        if self.plot_type == "dynamite":
            return SimpleDynamitePlot
        raise ValueError("plot_type must be 'dynamite', 'box', 'hist', 'cdf', or 'line'")

    def _plot_both(self, contact_df: pd.DataFrame, sum_df: pd.DataFrame):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("white")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})

        contact_plotter = self._plot_contacts(contact_df)
        contact_plotter.run(ax=axes[0])

        sum_plotter = self._plot_sums(sum_df)
        sum_plotter.run(ax=axes[1])

        max_sum = sum_df["sum_amplitude"].max()
        if pd.notna(max_sum):
            ylim_top = max_sum * 1.05
            axes[0].set_ylim(top=ylim_top)
            axes[1].set_ylim(top=ylim_top)

        fig.tight_layout()
        out_dir = self._resolve_out_dir()
        if self.plot_type == "box":
            subdir = "box_plots"
            fname = "stim_params_boxplot.svg"
        elif self.plot_type == "hist":
            subdir = "hist_plots"
            fname = "stim_params_hist.svg"
        elif self.plot_type == "cdf":
            subdir = "cdf_plots"
            fname = "stim_params_cdf.svg"
        elif self.plot_type == "line":
            subdir = "line_plots"
            fname = "stim_params_lineplot.svg"
        else:
            subdir = "dynamite_plots"
            fname = "stim_params_dynamite.svg"
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
        fig.savefig(
            os.path.join(out_dir, f"{subdir}/{fname}"),
            bbox_inches="tight",
        )
        plt.show()

    def _plot_contacts_hist(self, contact_df: pd.DataFrame):
        title = self.contact_plot_title or "Contact amplitudes"
        wide = contact_df.pivot_table(index="electrode", columns="contact", values="amplitude")
        y_label = self.contact_y_label
        if y_label == "Amplitude (mA)":
            y_label = "Count"
        plotter = SimpleHistogramPlot(
            wide.reset_index(drop=True),
            value_cols=[c for c in wide.columns],
            dataset_name=title,
            out_dir=None,
            xlabel=self.contact_x_label or "Amplitude (mA)",
            ylabel=y_label or "Count",
        )
        return plotter

    def _plot_sums_hist(self, sum_df: pd.DataFrame):
        title = self.sum_plot_title or "Summed amplitude"
        y_label = self.sum_y_label
        if y_label == "Amplitude (mA)":
            y_label = "Count"
        plotter = SimpleHistogramPlot(
            sum_df,
            value_cols=["sum_amplitude"],
            dataset_name=title,
            out_dir=None,
            xlabel=self.sum_x_label or "Amplitude (mA)",
            ylabel=y_label or "Count",
        )
        return plotter

    def _plot_contacts_cdf(self, contact_df: pd.DataFrame):
        title = self.contact_plot_title or "Contact amplitudes"
        wide = contact_df.pivot_table(index="electrode", columns="contact", values="amplitude")
        y_label = self.contact_y_label
        if y_label == "Amplitude (mA)":
            y_label = "Cumulative Density"
        plotter = SimpleCdfPlot(
            wide.reset_index(drop=True),
            value_cols=[c for c in wide.columns],
            dataset_name=title,
            out_dir=None,
            xlabel=self.contact_x_label or "Amplitude (mA)",
            ylabel=y_label or "Cumulative Density",
        )
        return plotter

    def _plot_sums_cdf(self, sum_df: pd.DataFrame):
        title = self.sum_plot_title or "Summed amplitude"
        y_label = self.sum_y_label
        if y_label == "Amplitude (mA)":
            y_label = "Cumulative Density"
        plotter = SimpleCdfPlot(
            sum_df,
            value_cols=["sum_amplitude"],
            dataset_name=title,
            out_dir=None,
            xlabel=self.sum_x_label or "Amplitude (mA)",
            ylabel=y_label or "Cumulative Density",
        )
        return plotter

    def _plot_contacts_line(self, contact_df: pd.DataFrame):
        title = self.contact_plot_title or "Contact amplitudes"
        order = self._contact_order()
        plot_df = contact_df.copy()
        plot_df["contact"] = pd.Categorical(plot_df["contact"], categories=order, ordered=True)
        plotter = SimpleLinePlot(
            plot_df,
            x_col="contact",
            y_col="amplitude",
            hue_col=None,
            dataset_name=title,
            out_dir=None,
            xlabel=self.contact_x_label or "Contact",
            ylabel=self.contact_y_label or "Amplitude (mA)",
            sort_x=False,
        )
        return plotter

    def _plot_sums_line(self, sum_df: pd.DataFrame):
        title = self.sum_plot_title or "Summed amplitude"
        plotter = SimpleLinePlot(
            sum_df,
            x_col="electrode",
            y_col="sum_amplitude",
            hue_col=None,
            dataset_name=title,
            out_dir=None,
            xlabel=self.sum_x_label or "Electrode",
            ylabel=self.sum_y_label or "Amplitude (mA)",
        )
        return plotter


class EvalStimParamsPair(EvalStimParams):
    """
    Pairwise evaluation of stimulation parameters across two JSON columns.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        json_col_a: str,
        json_col_b: str,
        out_dir: str | None = None,
        subject_col: str | None = None,
        plot_type: Literal["box", "dynamite", "cdf", "pair_superiority"] = "box",
        group_labels: Iterable[str] | None = None,
        contact_labels: Iterable[str] | None = None,
        contact_plot_title: str | None = 'Distribution of Current Across Contacts',
        contact_x_label: str | None = 'Contact',
        contact_y_label: str | None = 'Amplitude (mA)',
        sum_plot_title: str | None = 'Overall Amplitude',
        sum_x_label: str | None = '',
        sum_y_label: str | None = 'Amplitude (mA)',
    ):
        super().__init__(
            csv_path=csv_path,
            json_col=json_col_a,
            out_dir=out_dir,
            subject_col=subject_col,
            plot_type=plot_type,
            contact_labels=contact_labels,
            contact_plot_title=contact_plot_title,
            contact_x_label=contact_x_label,
            contact_y_label=contact_y_label,
            sum_plot_title=sum_plot_title,
            sum_x_label=sum_x_label,
            sum_y_label=sum_y_label,
        )
        self.json_col_a = json_col_a
        self.json_col_b = json_col_b
        if group_labels is None:
            self.group_labels = [json_col_a, json_col_b]
        else:
            labels = list(group_labels)
            if len(labels) != 2:
                raise ValueError("group_labels must have exactly 2 entries.")
            self.group_labels = labels

    def run(self):
        self._load_csv()
        self._validate_columns_pair()
        arrays_a, arrays_b = self._load_json_arrays_pair()
        self._arrays = arrays_a + arrays_b
        self._validate_lengths(self._arrays)
        contact_df = self._build_contact_dataframe_pair(arrays_a, arrays_b)
        sum_df = self._build_sum_dataframe_pair(arrays_a, arrays_b)
        if self.plot_type == "pair_superiority":
            self._plot_pair_superiority(arrays_a, arrays_b)
        else:
            self._plot_both_pair(contact_df, sum_df)
        return contact_df, sum_df

    def _validate_columns_pair(self):
        for col in [self.json_col_a, self.json_col_b]:
            if col not in self._df.columns:
                raise ValueError(f"Missing required column: {col}")
        if self.subject_col and self.subject_col not in self._df.columns:
            raise ValueError(f"subject_col '{self.subject_col}' not found in CSV")

    def _load_json_arrays_pair(self):
        arrays_a = []
        arrays_b = []
        for _, row in self._df.iterrows():
            p_a = row[self.json_col_a]
            p_b = row[self.json_col_b]
            if not (isinstance(p_a, str) and p_a.strip() and isinstance(p_b, str) and p_b.strip()):
                print("Paired input required: missing JSON path in one of the pair columns. Dropping row.")
                continue
            arrays_a.append(self._read_json_list(p_a))
            arrays_b.append(self._read_json_list(p_b))
        if not arrays_a or not arrays_b:
            raise ValueError("No valid JSON arrays found in one or both columns.")
        if len(arrays_a) != len(arrays_b):
            raise ValueError("Paired input required: unequal number of rows between JSON columns.")
        return arrays_a, arrays_b

    def _build_contact_dataframe_pair(self, arrays_a, arrays_b) -> pd.DataFrame:
        rows = []
        for i, arr in enumerate(arrays_a):
            for c_idx, val in enumerate(arr):
                rows.append({
                    "electrode": i,
                    "group": self.group_labels[0],
                    "contact": self._contact_label(c_idx),
                    "amplitude": val,
                })
        for i, arr in enumerate(arrays_b):
            for c_idx, val in enumerate(arr):
                rows.append({
                    "electrode": i,
                    "group": self.group_labels[1],
                    "contact": self._contact_label(c_idx),
                    "amplitude": val,
                })
        return pd.DataFrame(rows)

    def _build_sum_dataframe_pair(self, arrays_a, arrays_b) -> pd.DataFrame:
        rows = []
        for i, arr in enumerate(arrays_a):
            rows.append({
                "electrode": i,
                "group": self.group_labels[0],
                "sum_amplitude": float(np.sum(arr)),
            })
        for i, arr in enumerate(arrays_b):
            rows.append({
                "electrode": i,
                "group": self.group_labels[1],
                "sum_amplitude": float(np.sum(arr)),
            })
        return pd.DataFrame(rows)

    def _plot_both_pair(self, contact_df: pd.DataFrame, sum_df: pd.DataFrame):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("white")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})

        self._plot_contacts_pair(contact_df).run(ax=axes[0])
        self._plot_sums_pair(sum_df).run(ax=axes[1])

        max_sum = sum_df["sum_amplitude"].max()
        if pd.notna(max_sum):
            ylim_top = max_sum * 1.05
            axes[0].set_ylim(top=ylim_top)
            axes[1].set_ylim(top=ylim_top)

        fig.tight_layout()
        out_dir = self._resolve_out_dir()
        if self.plot_type == "box":
            subdir = "box_plots"
            fname = "stim_params_pair_boxplot.svg"
        elif self.plot_type == "cdf":
            subdir = "cdf_plots"
            fname = "stim_params_pair_cdf.svg"
        elif self.plot_type == "pair_superiority":
            subdir = "pair_superiority_plots"
            fname = "stim_params_pair_superiority.svg"
        else:
            subdir = "dynamite_plots"
            fname = "stim_params_pair_dynamite.svg"
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{subdir}/{fname}"), bbox_inches="tight")
        plt.show()

    def _plot_contacts_pair(self, contact_df: pd.DataFrame):
        if self.plot_type == "pair_superiority":
            return None
        title = self.contact_plot_title or "Contact amplitudes"
        if self.plot_type == "cdf":
            return SimpleCdfPlotPair(
                contact_df,
                group_col="group",
                category_col="contact",
                value_col="amplitude",
                dataset_name=title,
                out_dir=None,
                xlabel=self.contact_x_label or "Amplitude (mA)",
                ylabel=self.contact_y_label or "Cumulative Density",
            )
        plotter_cls = SimpleDynamitePlotPair if self.plot_type == "dynamite" else SimpleBoxPlotPair
        return plotter_cls(
            contact_df,
            group_col="group",
            category_col="contact",
            value_col="amplitude",
            dataset_name=title,
            out_dir=None,
            xlabel=self.contact_x_label or "Contact",
            ylabel=self.contact_y_label or "Amplitude (mA)",
            order=self._contact_order(),
            hue_order=self.group_labels,
        )

    def _plot_sums_pair(self, sum_df: pd.DataFrame):
        if self.plot_type == "pair_superiority":
            return None
        title = self.sum_plot_title or "Summed amplitude"
        if self.plot_type == "cdf":
            return SimpleCdfPlotPair(
                sum_df,
                group_col="group",
                category_col=None,
                value_col="sum_amplitude",
                dataset_name=title,
                out_dir=None,
                xlabel=self.sum_x_label or "Amplitude (mA)",
                ylabel=self.sum_y_label or "Cumulative Density",
            )
        plotter_cls = SimpleDynamitePlotPair if self.plot_type == "dynamite" else SimpleBoxPlotPair
        return plotter_cls(
            sum_df,
            group_col="group",
            category_col=None,
            value_col="sum_amplitude",
            dataset_name=title,
            out_dir=None,
            xlabel=self.sum_x_label or "",
            ylabel=self.sum_y_label or "Amplitude (mA)",
            hue_order=self.group_labels,
        )

    def _plot_pair_superiority(self, arrays_a, arrays_b):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.lines import Line2D

        n_contacts = self._n_contacts or len(arrays_a[0])
        sns.set_style("white")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]})

        # Left: per-contact paired dots with connecting lines on a shared axis
        ax_contacts = axes[0]
        offset = 0.15
        color_a = "#8E8E8E"  # model1 (json_col_a)
        color_b = "#211D1E"  # model2 (json_col_b)

        contact_labels = self._contact_order()
        x_centers = np.arange(n_contacts)

        for c_idx in range(n_contacts):
            vals_a = np.array([a[c_idx] for a in arrays_a], dtype=float)
            vals_b = np.array([b[c_idx] for b in arrays_b], dtype=float)
            if np.any(pd.isna(vals_a)) or np.any(pd.isna(vals_b)):
                raise ValueError("Paired input required: contact-wise data is sparse or contains NaNs.")

            x_a = np.full_like(vals_a, x_centers[c_idx] + offset, dtype=float)
            x_b = np.full_like(vals_b, x_centers[c_idx] - offset, dtype=float)

            # Lines connecting paired points
            for i in range(len(vals_a)):
                line_color = color_b if (vals_b[i] - vals_a[i]) >= 0 else color_a
                ax_contacts.plot(
                    [x_b[i], x_a[i]],
                    [vals_b[i], vals_a[i]],
                    color=line_color,
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=2,
                )

            ax_contacts.scatter(
                x_b,
                vals_b,
                color=color_b,
                edgecolors="white",
                linewidths=1.0,
                s=50,
                alpha=0.9,
                zorder=3,
            )
            ax_contacts.scatter(
                x_a,
                vals_a,
                color=color_a,
                edgecolors="white",
                linewidths=1.0,
                s=50,
                alpha=0.9,
                zorder=3,
            )

        ax_contacts.set_title(self.contact_plot_title or "Contact amplitudes", fontsize=20)
        ax_contacts.set_xlabel(self.contact_x_label or "Contact", fontsize=18)
        ax_contacts.set_ylabel(self.contact_y_label or "Amplitude (mA)", fontsize=18)
        ax_contacts.set_xticks(x_centers)
        ax_contacts.set_xticklabels(contact_labels)
        ax_contacts.tick_params(axis="both", labelsize=14)
        for spine in ax_contacts.spines.values():
            spine.set_linewidth(2)
        sns.despine(ax=ax_contacts)

        legend_elems = [
            Line2D([0], [0], marker='o', color='none', label=self.group_labels[1],
                   markerfacecolor=color_b, markeredgecolor='white', markersize=8),
            Line2D([0], [0], marker='o', color='none', label=self.group_labels[0],
                   markerfacecolor=color_a, markeredgecolor='white', markersize=8),
        ]
        ax_contacts.legend(handles=legend_elems, frameon=False, fontsize=12)

        # Overall sum
        sums_a = [float(np.sum(a)) for a in arrays_a]
        sums_b = [float(np.sum(b)) for b in arrays_b]
        if any(pd.isna(sums_a)) or any(pd.isna(sums_b)):
            raise ValueError("Paired input required: overall amplitude data is sparse or contains NaNs.")
        max_sum = max(max(sums_a), max(sums_b))
        plotter = PairSuperiorityPlot(
            stat_array_1=sums_a,
            stat_array_2=sums_b,
            model1_name=self.group_labels[0],
            model2_name=self.group_labels[1],
            stat="Overall Amplitude",
            out_dir=None,
            method="bootstrap",
        )
        ax = axes[1]
        plotter.plot_paired_slopes(ax)
        plotter.annotate_paired_slopes(ax)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ylim_top = max_sum * 1.05
        axes[0].set_ylim(top=ylim_top)
        axes[1].set_ylim(top=ylim_top)

        fig.tight_layout()
        out_dir = self._resolve_out_dir()
        os.makedirs(os.path.join(out_dir, "pair_superiority_plots"), exist_ok=True)
        fig.savefig(os.path.join(out_dir, "pair_superiority_plots/stim_params_pair_superiority.svg"), bbox_inches="tight")
        plt.show()
