from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot_returned_species_analysis(
        outputs: Mapping,
        log_scale: bool = True,
        colors: Sequence[str] | None = None,
        figsize: tuple[float, float] = (12, 8),
        line_width: float = 2.5,
        marker_size: float = 8,
        marker_alpha: float = 0.7,
        survived_line_width: float = 1.5,
        survived_alpha: float = 0.45,
        survived_color: str = "#7f7f7f",
        first_returned_color: str = "#4285FF",
        last_returned_color: str = "#25478F",
        low_count_region_alpha: float = 0.12,
        show_low_count_annotation: bool = True,
        abx_region_color: str = "#f8cece",
        abx_region_width: float = 1.0,
        abx_region_alpha: float = 1.0,
        timepoint_spread: float = 0.32,
        xticks_fontsize: float = 14,
        ylabel_fontsize: float = 16,
        ytick_fontsize: float = 14,
        tick_length: float | None = None,
        tick_width: float | None = None,
        legend_fontsize: float = 11,
        legend_loc: str = "upper center",
        legend_bbox_to_anchor: tuple[float, float] = (0.5, 1),
        show_legend: bool = True,
        top_x: int | None = None,
        path=None,
):
    """Plot final-follow-up returned taxa across the complete timeline.

    Positive relative abundances are plotted as circles connected by lines.
    Undetected samples are displayed as downward triangles at one shared
    visual floor below the smallest ``1 / N`` abundance across the complete
    timeline, and trajectory lines pass through that floor. The actual
    non-rarefied ``1 / N`` and ``3 / N`` boundaries continue to follow the
    sequencing depth of each sample.

    Parameters
    ----------
    outputs : mapping
        Output from ``ReturnedSpeciesAnalysis.run(subject_id)``. Both
        rarefied and non-rarefied outputs are supported.
    log_scale : bool, default=True
        Display relative abundance on a logarithmic y-axis when true.
    colors : sequence of str, optional
        Backward-compatible override for ``last_returned_color``; only the
        first supplied color is used.
    timepoint_spread : float, default=0.32
        Total horizontal spread used to keep overlapping taxon markers visible
        at each timepoint. Must be in the interval ``[0, 1)``.
    marker_alpha : float, default=0.7
        Opacity of the taxon trajectories and detected/undetected markers.
    survived_line_width : float, default=1.5
        Width of trajectories for survived taxa passing the configured
        baseline-to-ABX reduction threshold.
    survived_alpha : float, default=0.45
        Opacity of the reduced-survived-taxon trajectories.
    survived_color : str, default="#7f7f7f"
        Shared color for reduced-survived-taxon trajectories.
    first_returned_color : str, default="#4285FF"
        Shared color for taxa returning at the first post-ABX sample.
    last_returned_color : str, default="#25478F"
        Shared color for taxa returning at the last post-ABX sample.
    low_count_region_alpha : float, default=0.12
        Opacity of the grey region between the one-read and conditional
        detection boundaries.
    show_low_count_annotation : bool, default=True
        Show the sampling-sensitive low-count-region text and arrow when true.
    abx_region_color : str, default="#f8cece"
        Fill color of the vertical band marking the ABX timepoint.
    abx_region_width : float, default=1.0
        Total width (in x-axis units) of the vertical band centered on the
        ABX timepoint. Must be non-negative.
    abx_region_alpha : float, default=1.0
        Opacity of the ABX timepoint band.
    tick_length : float, optional
        Length of the physical x- and y-axis tick marks in points. When
        omitted, use Matplotlib's active style setting.
    tick_width : float, optional
        Width of the physical x- and y-axis tick marks in points. When
        omitted, use Matplotlib's active style setting.
    legend_loc : str, default="upper center"
        Matplotlib ``loc`` value for the legend: which corner/edge of the
        legend box is placed at ``legend_bbox_to_anchor``.
    legend_bbox_to_anchor : tuple of float, default=(0.5, 1)
        Axes-fraction ``(x, y)`` point that the legend is anchored to. Together
        with ``legend_loc`` this places the legend anywhere on (or outside)
        the figure, e.g. ``legend_loc="upper left", legend_bbox_to_anchor=(1, 1)``
        puts it just outside the top-right of the axes.
    show_legend : bool, default=True
        Show the figure legend when true.
    top_x : int, optional
        Plot only the ``top_x`` taxa in each trajectory category, ranked
        independently by relative abundance at the last follow-up. Categories
        containing fewer than ``top_x`` taxa are shown in full.
    path : path-like, optional
        Save the figure when provided; otherwise display it.

    Returns
    -------
    tuple
        The Matplotlib ``(figure, axes)`` pair.
    """
    if not isinstance(outputs, Mapping):
        raise TypeError("outputs must be a mapping returned by ReturnedSpeciesAnalysis.run().")

    required_keys = {
        "follow_up_times",
        "relative_abundances",
        "species_ids",
        "timeline_labels",
    }
    missing_keys = required_keys.difference(outputs)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"outputs is missing required field(s): {missing}.")

    relative_abundances = np.asarray(outputs["relative_abundances"], dtype=float)
    species_ids = np.asarray(outputs["species_ids"], dtype=object)
    timeline_labels = np.asarray(outputs["timeline_labels"], dtype=object)
    follow_up_times = np.asarray(outputs["follow_up_times"], dtype=float)
    reduced_survived_relative_abundances = np.asarray(
        outputs.get(
            "reduced_survived_relative_abundances",
            np.empty((0, timeline_labels.size)),
        ),
        dtype=float,
    )
    first_returned_relative_abundances = np.asarray(
        outputs.get(
            "first_returned_relative_abundances",
            np.empty((0, timeline_labels.size)),
        ),
        dtype=float,
    )

    if relative_abundances.ndim != 2:
        raise ValueError("outputs['relative_abundances'] must be a 2D array.")
    if relative_abundances.shape[0] != species_ids.size:
        raise ValueError("species_ids must have one value per abundance trajectory.")
    if relative_abundances.shape[1] != timeline_labels.size:
        raise ValueError("timeline_labels must have one value per trajectory timepoint.")
    if (
            reduced_survived_relative_abundances.ndim != 2
            or reduced_survived_relative_abundances.shape[1] != timeline_labels.size
    ):
        raise ValueError(
            "reduced_survived_relative_abundances must be a 2D array with "
            "one column per trajectory timepoint."
        )
    if (
            first_returned_relative_abundances.ndim != 2
            or first_returned_relative_abundances.shape[1] != timeline_labels.size
    ):
        raise ValueError(
            "first_returned_relative_abundances must be a 2D array with "
            "one column per trajectory timepoint."
        )
    if (
            relative_abundances.shape[0] == 0
            and reduced_survived_relative_abundances.shape[0] == 0
            and first_returned_relative_abundances.shape[0] == 0
    ):
        raise ValueError(
            "The selected subject has no first-returned, last-returned, or "
            "reduced survived taxa passing the configured threshold."
        )
    sequencing_depths = _resolve_sequencing_depths(outputs, timeline_labels.size)
    if follow_up_times.ndim != 1 or follow_up_times.size != timeline_labels.size - 2:
        raise ValueError("follow_up_times must have one value per post-ABX timepoint.")
    if (
            not np.all(np.isfinite(follow_up_times))
            or np.any(follow_up_times < 0)
            or np.any(np.diff(follow_up_times) <= 0)
    ):
        raise ValueError("follow_up_times must be finite, non-negative, and strictly increasing.")
    if not np.all(np.isfinite(relative_abundances)) or np.any(relative_abundances < 0):
        raise ValueError("relative_abundances must contain finite, non-negative values.")
    if (
            not np.all(np.isfinite(reduced_survived_relative_abundances))
            or np.any(reduced_survived_relative_abundances < 0)
    ):
        raise ValueError(
            "reduced_survived_relative_abundances must contain finite, non-negative values."
        )
    if (
            not np.all(np.isfinite(first_returned_relative_abundances))
            or np.any(first_returned_relative_abundances < 0)
    ):
        raise ValueError(
            "first_returned_relative_abundances must contain finite, non-negative values."
        )
    if not isinstance(log_scale, (bool, np.bool_)):
        raise TypeError("log_scale must be a boolean.")
    if not isinstance(show_legend, (bool, np.bool_)):
        raise TypeError("show_legend must be a boolean.")
    if not isinstance(show_low_count_annotation, (bool, np.bool_)):
        raise TypeError("show_low_count_annotation must be a boolean.")
    if (
            not isinstance(marker_alpha, (int, float))
            or not np.isfinite(marker_alpha)
            or not 0 <= marker_alpha <= 1
    ):
        raise ValueError("marker_alpha must be a finite number in [0, 1].")
    if (
            not isinstance(survived_alpha, (int, float))
            or not np.isfinite(survived_alpha)
            or not 0 <= survived_alpha <= 1
    ):
        raise ValueError("survived_alpha must be a finite number in [0, 1].")
    if (
            not isinstance(survived_line_width, (int, float))
            or not np.isfinite(survived_line_width)
            or survived_line_width <= 0
    ):
        raise ValueError("survived_line_width must be a positive finite number.")
    if (
            not isinstance(low_count_region_alpha, (int, float))
            or not np.isfinite(low_count_region_alpha)
            or not 0 <= low_count_region_alpha <= 1
    ):
        raise ValueError("low_count_region_alpha must be a finite number in [0, 1].")
    if (
            not isinstance(timepoint_spread, (int, float))
            or not np.isfinite(timepoint_spread)
            or not 0 <= timepoint_spread < 1
    ):
        raise ValueError("timepoint_spread must be a finite number in [0, 1).")
    if (
            not isinstance(abx_region_width, (int, float))
            or not np.isfinite(abx_region_width)
            or abx_region_width < 0
    ):
        raise ValueError("abx_region_width must be a non-negative finite number.")
    if (
            not isinstance(abx_region_alpha, (int, float))
            or not np.isfinite(abx_region_alpha)
            or not 0 <= abx_region_alpha <= 1
    ):
        raise ValueError("abx_region_alpha must be a finite number in [0, 1].")
    if (
            tick_length is not None
            and (
                isinstance(tick_length, (bool, np.bool_))
                or not isinstance(tick_length, (int, float))
                or not np.isfinite(tick_length)
                or tick_length < 0
            )
    ):
        raise ValueError("tick_length must be a non-negative finite number or None.")
    if (
            tick_width is not None
            and (
                isinstance(tick_width, (bool, np.bool_))
                or not isinstance(tick_width, (int, float))
                or not np.isfinite(tick_width)
                or tick_width <= 0
            )
    ):
        raise ValueError("tick_width must be a positive finite number or None.")
    if (
            top_x is not None
            and (
                isinstance(top_x, (bool, np.bool_))
                or not isinstance(top_x, (int, np.integer))
                or top_x <= 0
            )
    ):
        raise ValueError("top_x must be a positive integer or None.")

    if colors is not None:
        colors = list(colors)
        if not colors:
            raise ValueError("colors must contain at least one color when provided.")
        last_returned_color = colors[0]

    reduced_survived_relative_abundances, _ = _select_top_at_last_follow_up(
        reduced_survived_relative_abundances,
        top_x,
    )
    first_returned_relative_abundances, _ = _select_top_at_last_follow_up(
        first_returned_relative_abundances,
        top_x,
    )
    relative_abundances, last_returned_indices = _select_top_at_last_follow_up(
        relative_abundances,
        top_x,
    )
    species_ids = species_ids[last_returned_indices]
    # Match the survived-species plots: Baseline is at 0, ABX at 10, and
    # follow-ups are spaced by their actual number of days after ABX.
    x_values = np.concatenate(([0.0, 10.0], 10.0 + follow_up_times))
    detection_thresholds = 3.0 / sequencing_depths
    one_read_abundances = 1.0 / sequencing_depths
    undetected_display_value = float(np.min(one_read_abundances) * 0.5)
    undetected_display_values = np.full(
        timeline_labels.size,
        undetected_display_value,
    )
    boundaries_vary = not np.allclose(
        sequencing_depths,
        sequencing_depths[0],
        rtol=1e-12,
        atol=0,
    )
    non_rarefied = bool(outputs.get("non_rarefied", boundaries_vary))

    fig, ax = plt.subplots(figsize=figsize)
    abx_timepoint = 10.0
    ax.axvspan(
        abx_timepoint - abx_region_width / 2,
        abx_timepoint + abx_region_width / 2,
        color=abx_region_color,
        alpha=abx_region_alpha,
        zorder=-1,
    )
    if boundaries_vary:
        ax.fill_between(
            x_values,
            one_read_abundances,
            detection_thresholds,
            color="grey",
            alpha=low_count_region_alpha,
            zorder=0,
        )
    else:
        ax.axhspan(
            one_read_abundances[0],
            detection_thresholds[0],
            color="grey",
            alpha=low_count_region_alpha,
            zorder=0,
        )
    _plot_trajectory_group(
        ax,
        reduced_survived_relative_abundances,
        x_values,
        undetected_display_values,
        color=survived_color,
        line_width=survived_line_width,
        marker_size=marker_size,
        alpha=survived_alpha,
        timepoint_spread=timepoint_spread,
        zorder=2,
    )
    _plot_trajectory_group(
        ax,
        first_returned_relative_abundances,
        x_values,
        undetected_display_values,
        color=first_returned_color,
        line_width=line_width,
        marker_size=marker_size,
        alpha=marker_alpha,
        timepoint_spread=timepoint_spread,
        zorder=3,
    )
    _plot_trajectory_group(
        ax,
        relative_abundances,
        x_values,
        undetected_display_values,
        color=last_returned_color,
        line_width=line_width,
        marker_size=marker_size,
        alpha=marker_alpha,
        timepoint_spread=timepoint_spread,
        zorder=4,
        labels=species_ids,
    )

    if boundaries_vary:
        threshold_line, = ax.plot(
            x_values,
            detection_thresholds,
            color="black",
            linestyle="--",
            linewidth=2,
            zorder=1,
        )
        sampling_floor_line, = ax.plot(
            x_values,
            one_read_abundances,
            color="grey",
            linestyle="--",
            linewidth=2,
            zorder=1,
        )
    else:
        threshold_line = ax.axhline(
            detection_thresholds[0],
            color="black",
            linestyle="--",
            linewidth=2,
            zorder=1,
        )
        sampling_floor_line = ax.axhline(
            one_read_abundances[0],
            color="grey",
            linestyle="--",
            linewidth=2,
            zorder=1,
        )
    region_x = (x_values[0] + x_values[-1]) / 2
    region_floor = np.interp(region_x, x_values, one_read_abundances)
    region_threshold = np.interp(region_x, x_values, detection_thresholds)
    low_count_region_center = (
        np.sqrt(region_floor * region_threshold)
        if log_scale
        else (region_floor + region_threshold) / 2
    )
    annotation_y = (
        region_threshold * 1.8
        if log_scale
        else region_threshold + 1.25 * (region_threshold - region_floor)
    )
    if show_low_count_annotation:
        ax.annotate(
            "sampling-sensitive low-count region",
            xy=(region_x, low_count_region_center),
            xytext=(region_x, annotation_y),
            color="dimgray",
            fontsize=legend_fontsize,
            ha="center",
            va="bottom",
            arrowprops={
                "arrowstyle": "->",
                "color": "dimgray",
                "linewidth": 1.5,
            },
            zorder=5,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(
        timeline_labels,
        fontsize=xticks_fontsize,
        rotation=-90,
        color="black",
    )
    ax.tick_params(axis="x", colors="black")
    y_axis_label = "Relative abundance (log scale)" if log_scale else "Relative abundance"
    ax.set_ylabel(
        y_axis_label,
        fontsize=ylabel_fontsize,
        color="black",
    )
    ax.tick_params(axis="y", labelsize=ytick_fontsize, colors="black")
    tick_mark_options = {}
    if tick_length is not None:
        tick_mark_options["length"] = tick_length
    if tick_width is not None:
        tick_mark_options["width"] = tick_width
    if tick_mark_options:
        ax.tick_params(axis="both", **tick_mark_options)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=undetected_display_value * 0.75)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0e}"))
    else:
        ax.set_ylim(bottom=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(False)

    depth_symbol = "N" if non_rarefied else "N_0"
    undetected_handle = Line2D(
        [],
        [],
        color="black",
        marker="v",
        linestyle="None",
        markersize=marker_size,
        label=f"Undetected (shown below minimum 1/${depth_symbol}$)",
    )
    threshold_line.set_label(
        f"Upper boundary at 3/${depth_symbol}$: 95% conditional detection boundary"
    )
    sampling_floor_line.set_label(
        f"Lower boundary at 1/${depth_symbol}$: one-read abundance"
    )
    legend_handles = []
    if reduced_survived_relative_abundances.shape[0] > 0:
        survived_handle = Line2D(
            [],
            [],
            color=survived_color,
            linewidth=survived_line_width,
            marker="o",
            markersize=marker_size,
            alpha=survived_alpha,
            label=(
                "Survived species with >"
                f"{outputs.get('survived_reduction_threshold', 100):g}-fold ABX reduction"
            ),
        )
        legend_handles.append(survived_handle)
    if first_returned_relative_abundances.shape[0] > 0:
        legend_handles.append(Line2D(
            [],
            [],
            color=first_returned_color,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            alpha=marker_alpha,
            label="Early returning species",
        ))
    if relative_abundances.shape[0] > 0:
        legend_handles.append(Line2D(
            [],
            [],
            color=last_returned_color,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            alpha=marker_alpha,
            label="Late returning species",
        ))
    legend_handles.extend([
        threshold_line,
        sampling_floor_line,
        undetected_handle,
    ])
    if show_legend:
        legend = ax.legend(
            handles=legend_handles,
            fontsize=legend_fontsize,
            frameon=True,
            edgecolor="black",
            facecolor="white",
            bbox_to_anchor=legend_bbox_to_anchor,
            loc=legend_loc,
            ncol=1,
        )
        legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()

    return fig, ax


def _validate_sequencing_depth(value) -> float:
    try:
        sequencing_depth = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("outputs['sequencing_depth'] must be a positive finite number.") from error
    if not np.isfinite(sequencing_depth) or sequencing_depth <= 0:
        raise ValueError("outputs['sequencing_depth'] must be a positive finite number.")
    return sequencing_depth


def _resolve_sequencing_depths(outputs: Mapping, n_timepoints: int) -> np.ndarray:
    if "sequencing_depths" in outputs:
        sequencing_depths = np.asarray(outputs["sequencing_depths"], dtype=float)
    elif "sequencing_depth" in outputs:
        raw_depth = np.asarray(outputs["sequencing_depth"], dtype=float)
        if raw_depth.ndim == 0:
            sequencing_depths = np.full(
                n_timepoints,
                _validate_sequencing_depth(raw_depth),
            )
        else:
            sequencing_depths = raw_depth
    else:
        raise KeyError(
            "outputs is missing required field 'sequencing_depths' or 'sequencing_depth'."
        )

    if sequencing_depths.ndim != 1 or sequencing_depths.size != n_timepoints:
        raise ValueError(
            "sequencing_depths must have one value per trajectory timepoint."
        )
    if not np.all(np.isfinite(sequencing_depths)) or np.any(sequencing_depths <= 0):
        raise ValueError("sequencing_depths must contain positive finite values.")
    return sequencing_depths


def _select_top_at_last_follow_up(
        abundance_trajectories: np.ndarray,
        top_x: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n_trajectories = abundance_trajectories.shape[0]
    all_indices = np.arange(n_trajectories, dtype=int)
    if top_x is None or n_trajectories <= top_x:
        return abundance_trajectories, all_indices

    selected_indices = np.argsort(
        -abundance_trajectories[:, -1],
        kind="stable",
    )[:top_x]
    return abundance_trajectories[selected_indices], selected_indices


def _plot_trajectory_group(
        ax,
        abundance_trajectories: np.ndarray,
        x_values: np.ndarray,
        undetected_floors: np.ndarray,
        *,
        color: str,
        line_width: float,
        marker_size: float,
        alpha: float,
        timepoint_spread: float,
        zorder: float,
        labels: Sequence | None = None,
) -> None:
    offsets = _trajectory_offsets(
        abundance_trajectories.shape[0],
        timepoint_spread,
    )
    if labels is not None and len(labels) != abundance_trajectories.shape[0]:
        raise ValueError("labels must have one value per abundance trajectory.")

    for trajectory_index, (abundances, offset) in enumerate(zip(
            abundance_trajectories,
            offsets,
    )):
        detected = abundances > 0
        display_abundances = np.where(
            detected,
            abundances,
            undetected_floors,
        )
        species_x_values = x_values + offset
        label = (
            str(labels[trajectory_index])
            if labels is not None
            else "_nolegend_"
        )
        ax.plot(
            species_x_values,
            display_abundances,
            color=color,
            linewidth=line_width,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )
        if np.any(~detected):
            ax.scatter(
                species_x_values[~detected],
                undetected_floors[~detected],
                color=color,
                marker="v",
                s=marker_size ** 2,
                linewidths=0,
                alpha=alpha,
                zorder=zorder + 0.2,
            )
        if np.any(detected):
            ax.scatter(
                species_x_values[detected],
                abundances[detected],
                color=color,
                marker="o",
                s=marker_size ** 2,
                linewidths=0,
                alpha=alpha,
                zorder=zorder + 0.3,
            )


def _trajectory_offsets(n_trajectories: int, timepoint_spread: float) -> np.ndarray:
    if n_trajectories == 0:
        return np.empty(0, dtype=float)
    if n_trajectories == 1:
        return np.array([0.0])
    return np.linspace(
        -timepoint_spread / 2,
        timepoint_spread / 2,
        n_trajectories,
    )
