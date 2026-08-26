#!/usr/bin/env python3

"""Focused benchmark LOC across all four programming models.

Counts curated barebones excerpts under `refs/`, one file per
benchmark/variant.  Each excerpt keeps the model-specific data movement and
compute and drops the parts every variant pays equally: parameter plumbing,
timing instrumentation, host input synthesis, and result verification.

Migrated from ../../analyze_model_loc.py, which covered six benchmarks and
three C/C++ models; this adds Julia and the two newer benchmarks.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ANALYSIS_ROOT.parent
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "loc-analysis"

# The six the original comparison covered, kept as a named subset so the
# earlier figure stays reproducible after the suite grew.
LEGACY_BENCHMARKS = ("elementwise", "hist", "red", "linreg", "knn", "kmeans")
BENCHMARK_ORDER = LEGACY_BENCHMARKS + ("vector_search", "multitask_classifier")

VARIANT_ORDER = ("julia", "polymerpim", "simplepim", "baseline")
VARIANT_SUFFIX = {
    "julia": "jl",
    "polymerpim": "cc",
    "simplepim": "c",
    "baseline": "cc",
}

# Which pairs the report quotes a reduction for: (subject, reference).
COMPARISONS = (
    ("polymerpim", "simplepim"),
    ("polymerpim", "baseline"),
    ("julia", "simplepim"),
    ("julia", "baseline"),
    ("julia", "polymerpim"),
)


def ref_path(benchmark: str, variant: str) -> str:
    return f"refs/{benchmark}/{variant}.{VARIANT_SUFFIX[variant]}.ref"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count focused benchmark LOC from curated barebones reference "
            "files for Julia, PolymerPIM, SimplePIM, and baseline "
            "implementations."
        )
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARK_ORDER,
        default=BENCHMARK_ORDER,
        help="Subset of benchmarks to analyze.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=VARIANT_ORDER,
        help="Subset of variants to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for CSV/Markdown output (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def count_c_like_loc(lines: list[str]) -> tuple[int, int]:
    raw_loc = sum(1 for line in lines if line.strip())
    logical_loc = 0
    in_block_comment = False

    for line in lines:
        i = 0
        code = []
        while i < len(line):
            if in_block_comment:
                end = line.find("*/", i)
                if end == -1:
                    break
                in_block_comment = False
                i = end + 2
                continue
            if line.startswith("//", i):
                break
            if line.startswith("/*", i):
                in_block_comment = True
                i += 2
                continue
            code.append(line[i])
            i += 1
        if "".join(code).strip():
            logical_loc += 1

    return raw_loc, logical_loc


def count_julia_loc(lines: list[str]) -> tuple[int, int]:
    # Julia comments: `#` to end of line, nestable `#= ... =#`, and triple-
    # quoted docstrings. A `#` inside a string literal is code, so string
    # state is tracked rather than scanning for `#` blindly.
    raw_loc = sum(1 for line in lines if line.strip())
    logical_loc = 0
    block_depth = 0
    in_docstring = False

    for line in lines:
        i = 0
        code = []
        while i < len(line):
            if block_depth:
                if line.startswith("#=", i):
                    block_depth += 1
                    i += 2
                elif line.startswith("=#", i):
                    block_depth -= 1
                    i += 2
                else:
                    i += 1
                continue
            if in_docstring:
                end = line.find('"""', i)
                if end == -1:
                    break
                in_docstring = False
                i = end + 3
                continue
            if line.startswith('"""', i):
                in_docstring = True
                i += 3
                continue
            if line.startswith("#=", i):
                block_depth += 1
                i += 2
                continue
            if line[i] == "#":
                break
            if line[i] == '"':
                # Consume a single-line string so a `#` inside it survives.
                code.append(line[i])
                i += 1
                while i < len(line):
                    if line[i] == "\\":
                        code.append(line[i:i + 2])
                        i += 2
                        continue
                    code.append(line[i])
                    if line[i] == '"':
                        i += 1
                        break
                    i += 1
                continue
            code.append(line[i])
            i += 1
        if "".join(code).strip():
            logical_loc += 1

    return raw_loc, logical_loc


def count_loc(variant: str, lines: list[str]) -> tuple[int, int]:
    return (count_julia_loc(lines) if variant == "julia"
            else count_c_like_loc(lines))


def summarize(benchmarks, variants):
    summary_rows = []
    file_rows = []
    missing = []

    for benchmark in benchmarks:
        for variant in variants:
            rel_path = ref_path(benchmark, variant)
            path = ANALYSIS_ROOT / rel_path
            if not path.is_file():
                missing.append(rel_path)
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            raw_loc, logical_loc = count_loc(variant, lines)

            summary_rows.append({
                "benchmark": benchmark,
                "variant": variant,
                "logical_loc": logical_loc,
                "raw_loc": raw_loc,
                "source_file_count": 1,
            })
            file_rows.append({
                "benchmark": benchmark,
                "variant": variant,
                "path": rel_path,
                "logical_loc": logical_loc,
                "raw_loc": raw_loc,
            })

    if missing:
        raise SystemExit(
            "missing reference file(s):\n  " + "\n  ".join(missing))
    return summary_rows, file_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def variant_rows(summary_rows: list[dict], benchmark: str) -> dict[str, dict]:
    return {row["variant"]: row
            for row in summary_rows if row["benchmark"] == benchmark}


def pct_reduction(reference: int, subject: int) -> float:
    return 100.0 * (1.0 - (subject / reference))


def aggregate(summary_rows: list[dict], benchmarks) -> dict:
    """Per-benchmark mean reduction, plus the pooled-total reduction.

    The mean weights every benchmark equally; the pooled figure weights by
    size. They differ, so both are reported rather than one standing in for
    the other.
    """
    present = [b for b in benchmarks
               if b in {row["benchmark"] for row in summary_rows}]
    metrics: dict = {"benchmark_count": len(present)}
    if not present:
        return metrics

    loc = {b: variant_rows(summary_rows, b) for b in present}
    variants = {row["variant"] for row in summary_rows}

    for variant in VARIANT_ORDER:
        if variant in variants:
            metrics[f"total_{variant}_loc"] = sum(
                loc[b][variant]["logical_loc"] for b in present)

    for subject, reference in COMPARISONS:
        if not {subject, reference} <= variants:
            continue
        per_benchmark = [
            pct_reduction(loc[b][reference]["logical_loc"],
                          loc[b][subject]["logical_loc"])
            for b in present
        ]
        metrics[f"mean_{subject}_vs_{reference}_pct"] = (
            sum(per_benchmark) / len(per_benchmark))
        metrics[f"total_{subject}_vs_{reference}_pct"] = pct_reduction(
            sum(loc[b][reference]["logical_loc"] for b in present),
            sum(loc[b][subject]["logical_loc"] for b in present))
    return metrics


def metric_table(title: str, metrics: dict, variants) -> list[str]:
    lines = [f"### {title}", "",
             "| Metric | Value |", "| --- | ---: |",
             f"| Benchmark count | {metrics['benchmark_count']} |"]
    for variant in VARIANT_ORDER:
        key = f"total_{variant}_loc"
        if key in metrics:
            lines.append(f"| Total {variant} LOC | {metrics[key]} |")
    for subject, reference in COMPARISONS:
        key = f"mean_{subject}_vs_{reference}_pct"
        if key in metrics:
            lines.append(
                f"| Mean per-benchmark {subject} reduction vs {reference} "
                f"| {metrics[key]:.1f}% |")
    for subject, reference in COMPARISONS:
        key = f"total_{subject}_vs_{reference}_pct"
        if key in metrics:
            lines.append(
                f"| Pooled-total {subject} reduction vs {reference} "
                f"| {metrics[key]:.1f}% |")
    lines.append("")
    return lines


def write_report(output_dir: Path, summary_rows, file_rows, benchmarks,
                 variants):
    selected = [b for b in benchmarks
                if b in {row["benchmark"] for row in summary_rows}]
    legacy = [b for b in selected if b in LEGACY_BENCHMARKS]

    lines = [
        "# Focused LOC Comparison",
        "",
        "Counts come from curated barebones reference files under",
        "`loc-analysis/refs`. Logical LOC excludes blank lines and comments;",
        "each excerpt keeps only the model-specific data movement and compute.",
        "",
        "## Overall",
        "",
    ]
    lines.extend(metric_table(
        f"All {len(selected)} benchmarks", aggregate(summary_rows, selected),
        variants))
    if legacy and len(legacy) != len(selected):
        lines.extend(metric_table(
            f"Original {len(legacy)} benchmarks",
            aggregate(summary_rows, legacy), variants))

    for benchmark in benchmarks:
        rows = variant_rows(summary_rows, benchmark)
        if not rows:
            continue
        lines.extend([
            f"## {benchmark}",
            "",
            "| Variant | Logical LOC | Raw LOC | Reference file |",
            "| --- | ---: | ---: | --- |",
        ])
        for variant in VARIANT_ORDER:
            if variant not in rows:
                continue
            row = rows[variant]
            source = next(
                item["path"] for item in file_rows
                if item["benchmark"] == benchmark
                and item["variant"] == variant)
            lines.append(
                f"| {variant} | {row['logical_loc']} | {row['raw_loc']} "
                f"| `{source}` |")
        lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def print_stdout(summary_rows, benchmarks):
    print("benchmark,variant,logical_loc,raw_loc,source_file_count")
    for row in summary_rows:
        print(f"{row['benchmark']},{row['variant']},{row['logical_loc']},"
              f"{row['raw_loc']},{row['source_file_count']}")
    selected = [b for b in benchmarks
                if b in {row["benchmark"] for row in summary_rows}]
    metrics = aggregate(summary_rows, selected)
    print()
    for subject, reference in COMPARISONS:
        key = f"mean_{subject}_vs_{reference}_pct"
        if key in metrics:
            print(f"overall,mean_{subject}_vs_{reference}_pct="
                  f"{metrics[key]:.1f}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, file_rows = summarize(args.benchmarks, args.variants)
    selected = [b for b in args.benchmarks
                if b in {row["benchmark"] for row in summary_rows}]

    write_csv(output_dir / "summary.csv", summary_rows,
              ["benchmark", "variant", "logical_loc", "raw_loc",
               "source_file_count"])
    write_csv(output_dir / "file_breakdown.csv", file_rows,
              ["benchmark", "variant", "path", "logical_loc", "raw_loc"])

    overall = aggregate(summary_rows, selected)
    write_csv(output_dir / "overall_metrics.csv", [overall],
              list(overall.keys()))
    legacy = [b for b in selected if b in LEGACY_BENCHMARKS]
    if legacy:
        legacy_metrics = aggregate(summary_rows, legacy)
        write_csv(output_dir / "overall_metrics_legacy_six.csv",
                  [legacy_metrics], list(legacy_metrics.keys()))

    write_report(output_dir, summary_rows, file_rows, args.benchmarks,
                 args.variants)
    print_stdout(summary_rows, args.benchmarks)


if __name__ == "__main__":
    main()
