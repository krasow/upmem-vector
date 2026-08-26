#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

BENCHMARKS = Path(__file__).resolve().parent
CONFIG = BENCHMARKS / "main-benchmarks" / "benchmark.toml"
RUNS_CSV = BENCHMARKS / "results" / "runs.csv"
RESULTS = BENCHMARKS / "results"
SUMMARY_CSV = RESULTS / "weak-scaling-summary.csv"
ITERATION_FIGURE = RESULTS / "weak-scaling-mean-iteration.pdf"
RUNTIME_FIGURE = RESULTS / "weak-scaling-total-runtime.pdf"

BENCHMARK_ORDER = (
    "elementwise",
    "hist",
    "red",
    "kmeans",
    "knn",
    "linreg",
    "multitask_classifier",
    "vector_search",
)

EXCLUDED_BENCHMARKS = {"multitask_classifier"}

BENCHMARK_LABELS = {
    "elementwise": "Elementwise",
    "hist": "Histogram",
    "kmeans": "K-Means",
    "knn": "KNN",
    "linreg": "Linear Regression",
    "multitask_classifier": "Multitask Classifier",
    "red": "Reduction",
    "vector_search": "Vector Search",
}

VARIANT_ORDER = ("polymerpim", "julia", "baseline", "simplepim")
VARIANT_STYLES = {
    "polymerpim": ("PolymerPIM", "#3264a8", "o", "-"),
    "julia": ("Julia", "#dd7f27", "D", "-"),
    "baseline": ("Hand-tuned baseline", "#3b8f5a", "s", "-"),
    "simplepim": ("SimplePIM", "#b94a48", "^", "-"),
}


@dataclass(frozen=True)
class BenchmarkSelection:
    name: str
    elements_per_dpu: int
    dpus: tuple
    variants: tuple
    warmup: int
    iterations: int
    ntrials: int


@dataclass(frozen=True)
class Point:
    benchmark: str
    variant: str
    elements_per_dpu: int
    dpus: int
    ntrials: int
    mean_iteration_ms: float
    iteration_stddev_ms: float
    mean_runtime_s: float
    runtime_stddev_s: float


def load_selections():
    with CONFIG.open("rb") as file:
        config = tomllib.load(file)

    defaults = config["runner"]
    selections = []
    for name in BENCHMARK_ORDER:
        if name in EXCLUDED_BENCHMARKS:
            continue
        specs = config.get(name, [])
        if not specs:
            continue
        target_size = min(
            int(size)
            for spec in specs
            for size in spec["elements_per_dpu"]
        )
        spec = next(
            spec for spec in specs if target_size in spec["elements_per_dpu"]
        )
        selections.append(BenchmarkSelection(
            name=name,
            elements_per_dpu=target_size,
            dpus=tuple(int(value) for value in spec.get("dpus", defaults["dpus"])),
            variants=tuple(spec.get("variants", defaults["variants"])),
            warmup=int(spec.get("warmup", defaults["warmup"])),
            iterations=int(spec.get("iterations", defaults["iterations"])),
            ntrials=int(defaults["ntrials"]),
        ))
    return selections


def load_successful_rows():
    rows = []
    malformed = 0
    with RUNS_CSV.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if None in row:
                malformed += 1
                continue
            if row["status"] != "complete" or row["command_status"] != "success":
                continue
            if row["check"].lower() != "false":
                continue
            try:
                row["dpus"] = int(row["dpus"])
                row["elements_per_dpu"] = int(row["elements_per_dpu"])
                row["warmup"] = int(row["warmup"])
                row["iterations"] = int(row["iterations"])
                row["trial"] = int(row["trial"])
                row["time"] = float(row["time"])
                row["real_s"] = float(row["real_s"])
            except (KeyError, TypeError, ValueError):
                malformed += 1
                continue
            rows.append(row)
    if malformed:
        print(f"Ignored {malformed} malformed/incomplete CSV row(s)")
    return rows


def aggregate(rows, selections):
    selections_by_name = {selection.name: selection for selection in selections}
    latest_trials = {}
    for row in rows:
        selection = selections_by_name.get(row["benchmark"])
        if selection is None:
            continue
        if (row["elements_per_dpu"] != selection.elements_per_dpu
                or row["dpus"] not in selection.dpus
                or row["variant"] not in selection.variants
                or row["warmup"] != selection.warmup
                or row["iterations"] != selection.iterations
                or not 1 <= row["trial"] <= selection.ntrials):
            continue
        key = (row["benchmark"], row["variant"], row["dpus"], row["trial"])
        previous = latest_trials.get(key)
        if previous is None or row["timestamp"] > previous["timestamp"]:
            latest_trials[key] = row

    grouped = defaultdict(list)
    for (benchmark, variant, dpus, _trial), row in latest_trials.items():
        grouped[(benchmark, variant, dpus)].append(row)

    points = []
    complete_benchmarks = []
    for selection in selections:
        benchmark_points = []
        complete = True
        for variant in selection.variants:
            for dpus in selection.dpus:
                trials = grouped.get((selection.name, variant, dpus), [])
                if len(trials) != selection.ntrials:
                    complete = False
                    continue
                iteration_values = [row["time"] for row in trials]
                runtime_values = [row["real_s"] for row in trials]
                benchmark_points.append(Point(
                    benchmark=selection.name,
                    variant=variant,
                    elements_per_dpu=selection.elements_per_dpu,
                    dpus=dpus,
                    ntrials=len(trials),
                    mean_iteration_ms=average(iteration_values),
                    iteration_stddev_ms=sample_stddev(iteration_values),
                    mean_runtime_s=average(runtime_values),
                    runtime_stddev_s=sample_stddev(runtime_values),
                ))
        if complete:
            complete_benchmarks.append(selection.name)
            points.extend(benchmark_points)
        else:
            print(f"Omitting incomplete benchmark: {selection.name}")
    return points, complete_benchmarks


def average(values):
    return sum(values) / len(values)


def sample_stddev(values):
    if len(values) < 2:
        return 0.0
    mean = average(values)
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def geomean(values):
    if not values or any(value <= 0 for value in values):
        return math.nan
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate_speedup(points, candidate, reference, value):
    indexed = {
        (point.benchmark, point.variant, point.dpus): getattr(point, value)
        for point in points
    }
    benchmark_speedups = []
    for benchmark in BENCHMARK_ORDER:
        candidate_points = sorted(
            (point for point in points
             if point.benchmark == benchmark and point.variant == candidate),
            key=lambda point: point.dpus,
        )
        ratios = []
        for point in candidate_points:
            candidate_time = getattr(point, value)
            baseline_time = indexed.get((benchmark, "baseline", point.dpus))
            simplepim_time = indexed.get((benchmark, "simplepim", point.dpus))
            if reference == "baseline":
                reference_time = baseline_time
            elif reference == "simplepim":
                reference_time = simplepim_time
            else:
                available = [time for time in (baseline_time, simplepim_time)
                             if time is not None]
                reference_time = min(available) if available else None
            if (reference_time is not None and reference_time > 0
                    and candidate_time > 0):
                ratios.append(reference_time / candidate_time)
        if ratios:
            benchmark_speedups.append(geomean(ratios))
    return geomean(benchmark_speedups), len(benchmark_speedups)


def print_speedups(points, value, heading):
    print(f"{heading} (geomean across benchmarks):")
    for candidate, label in (("polymerpim", "PolymerPIM"),
                             ("julia", "Julia")):
        comparisons = []
        benchmark_count = 0
        for reference, reference_label in (
                ("baseline", "baseline"),
                ("simplepim", "SimplePIM"),
                ("best", "best of baseline and SimplePIM")):
            speedup, count = aggregate_speedup(
                points, candidate, reference, value)
            benchmark_count = max(benchmark_count, count)
            comparisons.append(f"{speedup:.3f}x vs {reference_label}")
        print(f"  {label} ({benchmark_count} benchmarks): "
              + ", ".join(comparisons))


def format_elements(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def configure_dpu_axis(axis, dpus):
    try:
        axis.set_xscale("log", base=2)
    except (TypeError, ValueError):
        axis.set_xscale("log", basex=2)
    axis.set_xticks(dpus)
    axis.set_xticklabels([str(value) for value in dpus])
    axis.minorticks_off()
    axis.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.8)


def plot_grid(points, benchmarks, value, error, ylabel, title, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = 3
    rows = math.ceil(len(benchmarks) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(10, rows * 2.75),
                                squeeze=False)
    flat_axes = axes.ravel()

    for index, (axis, benchmark) in enumerate(zip(flat_axes, benchmarks)):
        selected = [point for point in points if point.benchmark == benchmark]
        dpus = sorted({point.dpus for point in selected})
        elements_per_dpu = selected[0].elements_per_dpu
        for variant in VARIANT_ORDER:
            series = sorted(
                (point.dpus, getattr(point, value))
                for point in selected if point.variant == variant
            )
            if not series:
                continue
            label, color, marker, linestyle = VARIANT_STYLES[variant]
            x, y = zip(*series)
            yerr = [
                getattr(point, error)
                for point in sorted(
                    (point for point in selected if point.variant == variant),
                    key=lambda point: point.dpus,
                )
            ]
            axis.errorbar(
                x, y, yerr=yerr, color=color, marker=marker,
                linestyle=linestyle, linewidth=1.8, markersize=5.5,
                capsize=2.5, label=label,
            )

        axis.set_title(
            f"{BENCHMARK_LABELS[benchmark]}\n"
            f"{format_elements(elements_per_dpu)} elements/DPU",
            fontsize=11, fontweight="bold",
        )
        configure_dpu_axis(axis, dpus)
        axis.set_xlabel("DPUs")
        if index % columns == 0:
            axis.set_ylabel(ylabel)
        axis.margins(y=0.12)

    for axis in flat_axes[len(benchmarks):]:
        axis.set_visible(False)

    handles = [
        Line2D([0], [0], color=color, marker=marker, linestyle=linestyle,
               linewidth=1.8, markersize=5.5, label=label)
        for variant in VARIANT_ORDER
        for label, color, marker, linestyle in [VARIANT_STYLES[variant]]
    ]
    figure.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    figure.legend(handles=handles, loc="upper center", ncol=len(handles),
                  frameon=False, bbox_to_anchor=(0.5, 0.955))
    figure.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.35, w_pad=1.0)
    figure.savefig(output)
    plt.close(figure)


def write_summary(points):
    with SUMMARY_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "benchmark", "variant", "elements_per_dpu", "dpus", "ntrials",
            "mean_iteration_ms", "iteration_stddev_ms",
            "mean_runtime_s", "runtime_stddev_s",
        ])
        writer.writeheader()
        for point in points:
            writer.writerow({
                "benchmark": point.benchmark,
                "variant": point.variant,
                "elements_per_dpu": point.elements_per_dpu,
                "dpus": point.dpus,
                "ntrials": point.ntrials,
                "mean_iteration_ms": f"{point.mean_iteration_ms:.6f}",
                "iteration_stddev_ms": f"{point.iteration_stddev_ms:.6f}",
                "mean_runtime_s": f"{point.mean_runtime_s:.6f}",
                "runtime_stddev_s": f"{point.runtime_stddev_s:.6f}",
            })


def main():
    if not RUNS_CSV.is_file():
        raise SystemExit(f"missing benchmark results: {RUNS_CSV}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    points, benchmarks = aggregate(load_successful_rows(), load_selections())
    if not benchmarks:
        raise SystemExit("no complete benchmark grids found")

    write_summary(points)
    plot_grid(
        points, benchmarks, "mean_iteration_ms", "iteration_stddev_ms",
        "Mean iteration time (ms)",
        "Weak scaling: mean iteration time", ITERATION_FIGURE,
    )
    plot_grid(
        points, benchmarks, "mean_runtime_s", "runtime_stddev_s",
        "Mean process runtime (s)",
        "Weak scaling: end-to-end runtime", RUNTIME_FIGURE,
    )
    print(f"Averaged {points[0].ntrials} trials per data point")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print_speedups(points, "mean_iteration_ms", "Mean iteration speedup")
    print_speedups(points, "mean_runtime_s", "End-to-end runtime speedup")
    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {ITERATION_FIGURE}")
    print(f"Wrote {RUNTIME_FIGURE}")


if __name__ == "__main__":
    main()
