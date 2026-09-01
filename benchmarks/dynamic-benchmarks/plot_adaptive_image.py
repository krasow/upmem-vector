#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from statistics import median
from dataclasses import dataclass
from pathlib import Path

from plot_common import (configure_axis, draw_series, legend_handles,
                         load_pyplot, write_summary)


BENCHMARKS = Path(__file__).resolve().parent.parent
RESULTS = BENCHMARKS / "results" / "dynamic"
RUNS_CSV = RESULTS / "adaptive-image.csv"
SUMMARY_CSV = RESULTS / "adaptive-image-summary.csv"
FIGURE = RESULTS / "adaptive-image.pdf"
TRACE_TXT = RESULTS / "adaptive-image-trace.txt"

MODEL_ORDER = (
    "polymerpim-jit",
    "polymerpim-hybrid",
    "polymerpim-pipeline",
    "polymerpim-eager",
)
@dataclass(frozen=True)
class Summary:
    model: str
    elements_per_dpu: int
    dpus: int
    total_elements: int
    iterations: int
    ntrials: int
    mean_ms: float
    stddev_ms: float
    min_ms: float
    max_ms: float
    max_mean_ms: float
    max_min_ms: float
    max_max_ms: float
    wall_mean_s: float
    wall_stddev_s: float
    wall_min_s: float
    wall_max_s: float


def sample_stddev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values)
                     / (len(values) - 1))


def pooled_stats(rows):
    count = sum(int(row["iterations"]) for row in rows)
    mean = sum(float(row["time"]) * int(row["iterations"]) for row in rows) / count
    variance = sum(
        (int(row["iterations"]) - 1) * float(row["stddev"]) ** 2
        + int(row["iterations"]) * (float(row["time"]) - mean) ** 2
        for row in rows
    ) / (count - 1)
    return mean, math.sqrt(variance)


def row_identity(row):
    return tuple(row[name] for name in (
        "variant", "dpus", "elements_per_dpu", "warmup", "iterations",
        "trial", "check", "seed", "parameters",
    ))


def load_rows():
    completed = {}
    with RUNS_CSV.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["benchmark"] == "adaptive_image" and row["status"] == "complete":
                completed[row_identity(row)] = row

    rows = list(completed.values())
    signatures = {(row["iterations"], row["check"], row["parameters"])
                  for row in rows}
    if len(signatures) > 1:
        raise SystemExit("results contain mixed adaptive_image configurations; "
                         "rerun with --reset")
    return rows


def load_traces():
    # Per-iteration times from the focused 64-DPU runs; absent is fine.
    collected = defaultdict(list)
    if not TRACE_TXT.is_file():
        return {}
    for line in TRACE_TXT.read_text().splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[3].startswith("iteration_ms="):
            continue
        series = [float(value) for value
                  in fields[3].split("=", 1)[1].split(";") if value]
        if series:
            collected[(f"polymerpim-{fields[0]}", int(fields[1]))].append(series)
    # Median per iteration index, so one unlucky trial cannot set the shape.
    traces = {}
    for key, trials in collected.items():
        width = min(len(series) for series in trials)
        traces[key] = [median(series[index] for series in trials)
                       for index in range(width)]
    return traces


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], int(row["elements_per_dpu"]),
                 int(row["dpus"]))].append(row)

    summaries = []
    for (model, elements_per_dpu, dpus), trials in sorted(grouped.items()):
        mean_ms, stddev_ms = pooled_stats(trials)
        trial_maxima = [float(row["max"]) for row in trials]
        wall = [float(row["real_s"]) for row in trials if row["real_s"]]
        summaries.append(Summary(
            model=model,
            elements_per_dpu=elements_per_dpu,
            dpus=dpus,
            total_elements=elements_per_dpu * dpus,
            iterations=int(trials[0]["iterations"]),
            ntrials=len(trials),
            mean_ms=mean_ms,
            stddev_ms=stddev_ms,
            min_ms=min(float(row["min"]) for row in trials),
            max_ms=max(float(row["max"]) for row in trials),
            max_mean_ms=sum(trial_maxima) / len(trial_maxima),
            max_min_ms=min(trial_maxima),
            max_max_ms=max(trial_maxima),
            wall_mean_s=sum(wall) / len(wall),
            wall_stddev_s=sample_stddev(wall),
            wall_min_s=min(wall),
            wall_max_s=max(wall),
        ))
    return summaries


def draw(axis, rows, model, value, low=None, high=None):
    points = sorted((row for row in rows if row.model == model),
                    key=lambda row: row.dpus)
    if not points:
        return
    values = [getattr(row, value) for row in points]
    yerr = None
    if low and high:
        yerr = (
            [center - getattr(row, low) for center, row in zip(values, points)],
            [getattr(row, high) - center for center, row in zip(values, points)],
        )
    draw_series(axis, model, list(zip((row.dpus for row in points), values)),
                yerr=yerr, linewidth=2.1)


def plot(rows):
    plt = load_pyplot(FIGURE)

    sizes = sorted({row.elements_per_dpu for row in rows})
    if len(sizes) != 2:
        raise SystemExit("adaptive_image plot expects exactly two problem sizes")

    traces = load_traces()
    rows_count = 3 if traces else 2
    figure, axes = plt.subplots(rows_count, len(sizes),
                                figsize=(10, 3.2 * rows_count))
    for column, size in enumerate(sizes):
        selected = [row for row in rows if row.elements_per_dpu == size]
        column_axes = list(axes[:, column])
        trace_axis = column_axes.pop(0) if traces else None
        mean_axis, worst_axis = column_axes
        for model in MODEL_ORDER:
            draw(mean_axis, selected, model, "mean_ms")
            draw(worst_axis, selected, model,
                 "max_mean_ms", "max_min_ms", "max_max_ms")
            if trace_axis is not None:
                draw_series(trace_axis, model,
                            list(enumerate(traces.get((model, size), []),
                                           start=1)),
                            linewidth=1.6)

        top_axis = trace_axis if trace_axis is not None else mean_axis
        top_axis.set_title(f"{size:,} elements/DPU", fontweight="bold")
        mean_axis.set_ylabel("Mean iteration (ms)" if column == 0 else "")
        worst_axis.set_ylabel(
            "Worst iteration: mean, min-max (ms)" if column == 0 else "")
        if trace_axis is not None:
            trace_axis.set_ylabel(
                "Iteration time, 64 DPUs (ms)" if column == 0 else "")
            trace_axis.set_xlabel("Iteration")
            trace_axis.grid(True, which="major", color="#d8d8d8",
                            linewidth=0.8)
            trace_axis.set_axisbelow(True)
        dpus = sorted({row.dpus for row in selected})
        for axis in (mean_axis, worst_axis):
            configure_axis(axis, dpus, dpus, "DPUs")

    figure.suptitle("Adaptive image: dynamic execution trade-offs",
                    fontsize=15, fontweight="bold")
    figure.legend(handles=legend_handles(MODEL_ORDER, linewidth=2.1),
                  loc="upper center", ncol=4, frameon=False,
                  bbox_to_anchor=(0.5, 0.955))
    figure.tight_layout(rect=(0, 0, 1, 0.925), h_pad=1.6, w_pad=1.25)
    figure.savefig(FIGURE)


def main():
    if not RUNS_CSV.is_file():
        raise SystemExit("run adaptive_image.sh before plotting")
    rows = summarize(load_rows())
    if not rows:
        raise SystemExit("no complete adaptive_image measurements found")
    write_summary(SUMMARY_CSV, rows)
    plot(rows)
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
