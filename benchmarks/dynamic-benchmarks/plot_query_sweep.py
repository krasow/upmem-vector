#!/usr/bin/env python3

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results/query-sweep.csv"
SECTIONS = ROOT / "results/query-sweep.sections.csv"
SUMMARY = ROOT / "results/query-sweep-summary.csv"
FIGURE = ROOT / "results/query-sweep.svg"
MODELS = ("polymerpim-jit", "simplepim")
QUERY_OPS = 2


def parameters(value):
    result = {}
    for field in value.split(";"):
        if "=" in field:
            key, raw = field.split("=", 1)
            result[key] = int(raw) if raw.isdigit() else raw
    return result


def mean(values):
    return sum(values) / len(values)


def load_measurements():
    sections = {}
    with SECTIONS.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["benchmark"] != "dynamic_query":
                continue
            key = (row["timestamp"], row["invocation"], row["variant"],
                   row["trial"])
            sections.setdefault(key, {})[row["section"]] = float(row["time_ms"])

    grouped = defaultdict(lambda: defaultdict(list))
    with RUNS.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["benchmark"] != "dynamic_query" or row["status"] != "complete":
                continue
            params = parameters(row["parameters"])
            if int(params["query_ops"]) != QUERY_OPS:
                continue
            key = (row["timestamp"], row["invocation"], row["variant"],
                   row["trial"])
            measured = sections.get(key, {})
            if "query_first" not in measured or "query_reuse" not in measured:
                continue
            group = (row["variant"], int(row["total_elements"]),
                     int(params["query_ops"]))
            grouped[group]["first"].append(measured["query_first"])
            grouped[group]["reuse"].append(measured["query_reuse"])
            grouped[group]["query"].append(float(row["time"]))
    return {key: {name: mean(values) for name, values in metrics.items()}
            for key, metrics in grouped.items()}


def summarize(measurements):
    rows = []
    points = sorted({(elements, ops) for _, elements, ops in measurements})
    for elements, ops in points:
        pipeline = measurements.get(("polymerpim-pipeline", elements, ops))
        if not pipeline:
            continue
        pipeline_ms = pipeline["reuse"]
        for model in MODELS:
            result = measurements.get((model, elements, ops))
            if not result:
                continue
            compile_ms = max(0.0, result["first"] - result["reuse"])
            savings_ms = pipeline_ms - result["reuse"]
            break_even = compile_ms / savings_ms if savings_ms > 0 else math.inf
            rows.append({
                "model": model,
                "total_elements": elements,
                "query_ops": ops,
                "compile_ms": compile_ms,
                "pipeline_batch_ms": pipeline_ms,
                "compiled_batch_ms": result["reuse"],
                "break_even_batches": break_even,
                "measured_query_ms": result["query"],
            })
    return rows


def write_summary(rows):
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with SUMMARY.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: f"{value:.3f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def plot(rows, measurements):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise SystemExit(f"matplotlib is required to write {FIGURE}: {error}")

    models = {
        "polymerpim-jit": ("Blocking JIT", "#3264a8", "o"),
        "polymerpim-hybrid": ("Hybrid", "#dd7f27", "D"),
        "polymerpim-pipeline": ("Pipeline", "#3b8f5a", "s"),
        "simplepim": ("SimplePIM", "#b94a48", "^"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    def draw(axis, model, points):
        label, color, marker = models[model]
        points = sorted(points)
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color, marker=marker, linewidth=2.2, markersize=6,
                label=label,
            )

    for model in ("polymerpim-jit", "simplepim"):
        selected = [row for row in rows if row["model"] == model]
        draw(axes[0], model,
             [(row["total_elements"], row["compile_ms"])
              for row in selected])
        draw(axes[1], model,
             [(row["total_elements"], row["break_even_batches"])
              for row in selected if math.isfinite(row["break_even_batches"])])

    for model in models:
        draw(axes[2], model,
             [(key[1], value["query"])
              for key, value in measurements.items() if key[0] == model])

    element_counts = sorted({key[1] for key in measurements})
    tick_labels = [
        f"{value / 1_000_000:.3g}M" if value >= 1_000_000
        else f"{value / 1_000:.3g}K" for value in element_counts
    ]
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(element_counts)
        axis.set_xticklabels(tick_labels)
        axis.set_xlabel("Total elements")
        axis.grid(True, which="major", color="#d8d8d8", linewidth=0.8)
        axis.grid(True, which="minor", color="#eeeeee", linewidth=0.5)

    axes[0].set_title("Compilation overhead", fontweight="bold")
    axes[0].set_ylabel("First-batch overhead (ms)")
    axes[1].set_title("Break-even point", fontweight="bold")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Batches to beat pipeline")
    axes[1].axhline(1, color="#777777", linewidth=1)
    axes[2].set_title("Measured query latency", fontweight="bold")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Five-batch query latency (ms)")
    legend = [Line2D(
        [0], [0], color=color, marker=marker, linewidth=2.2,
        markersize=6, label=label,
    ) for label, color, marker in models.values()]
    fig.suptitle("Dynamic query compilation trade-offs", fontsize=15,
                 fontweight="bold")
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.5)
    fig.savefig(FIGURE)


def main():
    if not RUNS.is_file() or not SECTIONS.is_file():
        raise SystemExit("run query_sweep.sh before plotting")
    measurements = load_measurements()
    rows = summarize(measurements)
    if not rows:
        raise SystemExit("no complete dynamic_query measurements found")
    write_summary(rows)
    plot(rows, measurements)
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {FIGURE}")


if __name__ == "__main__":
    main()
