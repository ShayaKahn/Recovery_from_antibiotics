from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _coerce_boolean(values: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)

    converted = values.map({
        True: True,
        False: False,
        1: True,
        0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
    })
    if converted.isna().any():
        raise ValueError(f"{column_name} must contain boolean values.")
    return converted.astype(bool)


def calculate_reference_subsampling_plot_statistics(
        results: pd.DataFrame,
        alpha: float = 0.05,
        include_confidence_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize rank-1 significant subjects across subsampling repeats."""
    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    if results.empty:
        raise ValueError("results must contain at least one row.")

    if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be a number in the interval (0, 1).")
    if not isinstance(include_confidence_only, bool):
        raise TypeError("include_confidence_only must be a boolean value.")

    required_columns = {
        "subset_size",
        "repeat",
        "rank",
        "is_significant",
        "full_rank",
        "full_is_significant",
    }
    confidence_columns = {"adjusted_p_value", "full_adjusted_p_value"}
    if include_confidence_only:
        required_columns.update(confidence_columns)
    missing_columns = required_columns.difference(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"results is missing required column(s): {missing}.")

    plot_data = results.loc[:, list(required_columns)].copy()
    boolean_columns = {"is_significant", "full_is_significant"}
    numeric_columns = required_columns.difference(boolean_columns)
    for column in numeric_columns:
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
        values = plot_data[column].to_numpy(dtype=float)
        if column in confidence_columns:
            valid = np.isnan(values) | np.isfinite(values)
        else:
            valid = np.isfinite(values)
        if not np.all(valid):
            raise ValueError(f"{column} must contain only finite numeric values.")
    for column in boolean_columns:
        plot_data[column] = _coerce_boolean(plot_data[column], column)

    records = []
    for (subset_size, repeat), group in plot_data.groupby(
            ["subset_size", "repeat"],
            sort=True,
    ):
        if len(group) < 2:
            raise ValueError(
                "Each subset-size/repeat group must contain at least two subjects."
            )
        record = {
            "subset_size": int(subset_size),
            "repeat": int(repeat),
            "rank1_significant_fraction": (
                (group["rank"] == 1) & group["is_significant"]
            ).mean(),
            "full_rank1_significant_fraction": (
                (group["full_rank"] == 1) & group["full_is_significant"]
            ).mean(),
        }
        if include_confidence_only:
            record.update({
                "confidence_only_fraction": (
                    group["adjusted_p_value"] <= alpha
                ).mean(),
                "full_confidence_only_fraction": (
                    group["full_adjusted_p_value"] <= alpha
                ).mean(),
            })
        records.append(record)

    repeat_metrics = pd.DataFrame(records)
    full_fractions = repeat_metrics["full_rank1_significant_fraction"]
    if not np.allclose(full_fractions, full_fractions.iloc[0]):
        raise ValueError(
            "Full-reference rank-1 significant fraction must be identical "
            "across subset sizes and repeats."
        )
    if include_confidence_only:
        full_confidence_fractions = repeat_metrics[
            "full_confidence_only_fraction"
        ]
        if not np.allclose(
                full_confidence_fractions,
                full_confidence_fractions.iloc[0],
        ):
            raise ValueError(
                "Full-reference confidence-only fraction must be identical "
                "across subset sizes and repeats."
            )

    summary_rows = []
    for subset_size, group in repeat_metrics.groupby("subset_size", sort=True):
        values = group["rank1_significant_fraction"]
        row = {
            "subset_size": int(subset_size),
            "rank1_significant_fraction_median": float(values.median()),
            "rank1_significant_fraction_low": float(values.quantile(0.025)),
            "rank1_significant_fraction_high": float(values.quantile(0.975)),
            "full_rank1_significant_fraction": float(full_fractions.iloc[0]),
        }
        if include_confidence_only:
            confidence_values = group["confidence_only_fraction"]
            row.update({
                "confidence_only_fraction_median": float(
                    confidence_values.median()
                ),
                "confidence_only_fraction_low": float(
                    confidence_values.quantile(0.025)
                ),
                "confidence_only_fraction_high": float(
                    confidence_values.quantile(0.975)
                ),
                "full_confidence_only_fraction": float(
                    full_confidence_fractions.iloc[0]
                ),
            })
        summary_rows.append(row)

    return repeat_metrics, pd.DataFrame(summary_rows)


def plot_reference_subsampling_summary(
        results: pd.DataFrame,
        figsize: tuple[float, float] = (6, 4.5),
        color: str = "#2A9D8F",
        confidence_only_color: str = "#E76F51",
        show_confidence_only: bool = True,
        alpha: float = 0.05,
        path=None,
):
    """Plot the fraction of rank-1 significant subjects by reference size."""
    repeat_metrics, summary = calculate_reference_subsampling_plot_statistics(
        results,
        alpha=alpha,
        include_confidence_only=show_confidence_only,
    )

    x = summary["subset_size"].to_numpy(dtype=float)
    median = summary["rank1_significant_fraction_median"].to_numpy(dtype=float)
    low = summary["rank1_significant_fraction_low"].to_numpy(dtype=float)
    high = summary["rank1_significant_fraction_high"].to_numpy(dtype=float)
    full_fraction = float(summary["full_rank1_significant_fraction"].iloc[0])

    fig, ax = plt.subplots(figsize=figsize)
    x_margin = max(1.0, 0.04 * np.ptp(x))
    ax.fill_between(x, low, high, color=color, alpha=0.22)
    ax.plot(x, median, color=color, marker="o", linewidth=2.3)
    ax.axhline(
        full_fraction,
        color=color,
        linestyle="--",
        linewidth=1.2,
    )
    if show_confidence_only:
        confidence_median = summary[
            "confidence_only_fraction_median"
        ].to_numpy(dtype=float)
        confidence_low = summary[
            "confidence_only_fraction_low"
        ].to_numpy(dtype=float)
        confidence_high = summary[
            "confidence_only_fraction_high"
        ].to_numpy(dtype=float)
        full_confidence_fraction = float(
            summary["full_confidence_only_fraction"].iloc[0]
        )
        ax.fill_between(
            x,
            confidence_low,
            confidence_high,
            color=confidence_only_color,
            alpha=0.18,
        )
        ax.plot(
            x,
            confidence_median,
            color=confidence_only_color,
            marker="s",
            linewidth=2.3,
        )
        ax.axhline(
            full_confidence_fraction,
            color=confidence_only_color,
            linestyle="--",
            linewidth=1.2,
        )
    ax.set_xlabel("Number of reference baselines", fontsize=12)
    ylabel = "Fraction of subjects meeting the criterion"
    if not show_confidence_only:
        ylabel = "Fraction of subjects with rank 1\nand a significant result"
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
    ax.set_ylim(0, 1.04)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(False)

    legend_handles = [
        Patch(
            facecolor="0.5",
            edgecolor="0.5",
            alpha=0.2,
            label="95% resampling intervals",
        ),
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linewidth=2.3,
            label="Confidence + rank 1 criterion",
        ),
        Line2D(
            [0],
            [0],
            color=color,
            linestyle="--",
            linewidth=1.2,
            label="Confidence + rank 1 criterion: full reference",
        ),
    ]
    if show_confidence_only:
        legend_handles[2:2] = [
            Line2D(
                [0],
                [0],
                color=confidence_only_color,
                marker="s",
                linewidth=2.3,
                label="Confidence only criterion",
            ),
            Line2D(
                [0],
                [0],
                color=confidence_only_color,
                linestyle="--",
                linewidth=1.2,
                label="Confidence only criterion: full reference",
            ),
        ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        ncol=1,
        frameon=False,
    )
    fig.tight_layout()

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()

    return {
        "figure": fig,
        "axes": np.asarray([ax], dtype=object),
        "repeat_metrics": repeat_metrics,
        "summary": summary,
    }
