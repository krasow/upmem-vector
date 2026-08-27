#!/usr/bin/env python3

"""Focused benchmark LOC across all four programming models.

Counts curated barebones excerpts under `refs/`, one file per
benchmark/variant.  Each excerpt keeps the model-specific data movement and
compute and drops the parts every variant pays equally: parameter plumbing,
timing instrumentation, host input synthesis, and result verification.

Every file is counted twice: the built-in counter reports logical lines --
blank lines and comments dropped -- and boyter/scc contributes its own SLOC
definition plus complexity, cognitive complexity, and unique-line counts.  The
two SLOC figures agreeing is the check that neither counter is flattering
anyone; `scripts/install_scc.sh` provides scc.

Migrated from ../../analyze_model_loc.py, which covered six benchmarks and
three C/C++ models; this adds Julia and the two newer benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

ANALYSIS_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ANALYSIS_ROOT.parent
REPO_ROOT = BENCHMARK_DIR.parent
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "loc-analysis"
# scripts/install_scc.sh drops scc here, keeping the dependency inside the
# repo; anything already on PATH is the fallback.
VENDORED_SCC = REPO_ROOT / "opt" / "scc" / "bin" / "scc"

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

# Reference files end in `.ref` so they never compile by accident, which also
# hides the language from scc -- hence one invocation per language, each with
# `--count-as ref:<language>`.
SCC_LANGUAGE = {
    "julia": "Julia",
    "polymerpim": "C++",
    "simplepim": "C",
    "baseline": "C++",
}


class Column(NamedTuple):
    """One count: where scc reports it, what we call it, how to head it."""

    scc_key: str
    name: str
    heading: str


COLUMNS = (
    Column("", "logical_loc", "Logical LOC"),
    Column("Code", "scc_code", "scc SLOC"),
    Column("Lines", "scc_lines", "Lines"),
    Column("Comment", "scc_comment", "Comment"),
    Column("Blank", "scc_blank", "Blank"),
    Column("Complexity", "scc_complexity", "Complexity"),
    Column("Cognitive", "scc_cognitive", "Cognitive"),
    Column("Uloc", "scc_uloc", "ULOC"),
)
SCC_COLUMNS = tuple(column for column in COLUMNS if column.scc_key)


class MetricView(NamedTuple):
    """A count to reduce over, and the key namespace its figures live in."""

    column: str
    prefix: str
    title: str

    def key(self, name: str) -> str:
        return f"{self.prefix}{name}"


VIEWS = (
    MetricView("logical_loc", "", "logical LOC"),
    MetricView("scc_code", "scc_sloc_", "scc SLOC"),
    MetricView("scc_uloc", "scc_uloc_", "scc ULOC (distinct lines)"),
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
    parser.add_argument(
        "--scc-bin",
        default=str(VENDORED_SCC) if VENDORED_SCC.is_file() else "scc",
        help=(
            "scc executable to count with (default: <repo>/opt/scc/bin/scc "
            "if installed, else scc on PATH)."
        ),
    )
    return parser.parse_args()


def take_string_literal(line: str, start: int) -> tuple[str, int]:
    """Consume the literal opening at `start`, escapes included.

    Both strippers below need this: a comment marker inside a string is code,
    so the literal has to be stepped over rather than scanned into.
    """
    quote = line[start]
    text = [quote]
    i = start + 1
    while i < len(line):
        if line[i] == "\\":
            text.append(line[i:i + 2])
            i += 2
            continue
        text.append(line[i])
        i += 1
        if text[-1] == quote:
            break
    return "".join(text), i


def strip_c_comments(lines: Iterable[str]) -> Iterator[str]:
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
            elif line.startswith("//", i):
                break
            elif line.startswith("/*", i):
                in_block_comment = True
                i += 2
            elif line[i] in "\"'":
                text, i = take_string_literal(line, i)
                code.append(text)
            else:
                code.append(line[i])
                i += 1
        yield "".join(code)


def strip_julia_comments(lines: Iterable[str]) -> Iterator[str]:
    # Julia comments: `#` to end of line, nestable `#= ... =#`, and triple-
    # quoted docstrings.
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
            elif in_docstring:
                end = line.find('"""', i)
                if end == -1:
                    break
                in_docstring = False
                i = end + 3
            elif line.startswith('"""', i):
                in_docstring = True
                i += 3
            elif line.startswith("#=", i):
                block_depth += 1
                i += 2
            elif line[i] == "#":
                break
            elif line[i] == '"':
                text, i = take_string_literal(line, i)
                code.append(text)
            else:
                code.append(line[i])
                i += 1
        yield "".join(code)


def count_logical_loc(variant: str, lines: list[str]) -> int:
    strip = strip_julia_comments if variant == "julia" else strip_c_comments
    return sum(1 for line in strip(lines) if line.strip())


def scc_version(binary: str) -> str:
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True,
                              text=True, check=True)
    except FileNotFoundError:
        raise SystemExit(
            f"scc not found: {binary!r}.\n"
            "Install it into the repo with:\n"
            "  scripts/install_scc.sh")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{binary} --version failed: {exc.stderr.strip()}")
    return proc.stdout.strip().splitlines()[0]


def run_scc(binary: str, targets: list[tuple[str, Path]]) -> dict[Path, dict]:
    """Per-file scc counts, keyed by resolved path.

    `targets` pairs each file with the scc language to count it as; files are
    grouped so scc runs once per language instead of once per file.
    """
    by_language: dict[str, list[Path]] = defaultdict(list)
    for language, path in targets:
        by_language[language].append(path)

    counts: dict[Path, dict] = {}
    for language, group in sorted(by_language.items()):
        cmd = [binary, "--by-file", "--format", "json", "--no-cocomo",
               "--uloc", "--cognitive", "--count-as", f"ref:{language}",
               *(str(path) for path in group)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"scc failed for {language}: {exc.stderr.strip()}")
        for entry in json.loads(proc.stdout):
            for file_entry in entry.get("Files", []):
                # Cognitive and Uloc arrived in later scc releases; treat a
                # missing key as zero rather than failing the whole run.
                counts[Path(file_entry["Location"]).resolve()] = {
                    column.name: file_entry.get(column.scc_key, 0)
                    for column in SCC_COLUMNS
                }
    return counts


def summarize(benchmarks, variants, scc_binary: str) -> list[dict]:
    rows = []
    missing = []
    scc_targets = []

    for benchmark in benchmarks:
        for variant in variants:
            rel_path = ref_path(benchmark, variant)
            path = ANALYSIS_ROOT / rel_path
            if not path.is_file():
                missing.append(rel_path)
                continue
            rows.append({
                "benchmark": benchmark,
                "variant": variant,
                "path": rel_path,
                "logical_loc": count_logical_loc(
                    variant, path.read_text(encoding="utf-8").splitlines()),
            })
            scc_targets.append((SCC_LANGUAGE[variant], path))

    if missing:
        raise SystemExit(
            "missing reference file(s):\n  " + "\n  ".join(missing))

    scc_counts = run_scc(scc_binary, scc_targets)
    for row in rows:
        counts = scc_counts.get((ANALYSIS_ROOT / row["path"]).resolve())
        if counts is None:
            raise SystemExit(f"scc returned no counts for {row['path']}")
        row.update(counts)

    return rows


def selected_benchmarks(rows: list[dict], benchmarks) -> list[str]:
    counted = {row["benchmark"] for row in rows}
    return [b for b in benchmarks if b in counted]


def by_variant(rows: list[dict], benchmark: str) -> dict[str, dict]:
    return {row["variant"]: row
            for row in rows if row["benchmark"] == benchmark}


def pct_reduction(reference: int, subject: int) -> float:
    return 100.0 * (1.0 - (subject / reference))


def aggregate(rows: list[dict], benchmarks, view: MetricView) -> dict:
    """Per-benchmark mean reduction, plus the pooled-total reduction.

    The mean weights every benchmark equally; the pooled figure weights by
    size. They differ, so both are reported rather than one standing in for
    the other.
    """
    present = selected_benchmarks(rows, benchmarks)
    metrics: dict = {view.key("benchmark_count"): len(present)}
    if not present:
        return metrics

    count = {b: {variant: row[view.column]
                 for variant, row in by_variant(rows, b).items()}
             for b in present}
    variants = {row["variant"] for row in rows}

    for variant in VARIANT_ORDER:
        if variant in variants:
            metrics[view.key(f"total_{variant}_loc")] = sum(
                count[b][variant] for b in present)

    for subject, reference in COMPARISONS:
        if not {subject, reference} <= variants:
            continue
        per_benchmark = [
            pct_reduction(count[b][reference], count[b][subject])
            for b in present
        ]
        metrics[view.key(f"mean_{subject}_vs_{reference}_pct")] = (
            sum(per_benchmark) / len(per_benchmark))
        metrics[view.key(f"total_{subject}_vs_{reference}_pct")] = (
            pct_reduction(sum(count[b][reference] for b in present),
                          sum(count[b][subject] for b in present)))
    return metrics


def md_table(headings, table) -> str:
    """A markdown table: first column left-aligned, the rest right."""
    divider = ["---"] + ["---:"] * (len(headings) - 1)
    return "\n".join(
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in [headings, divider, *table])


def metric_table(title: str, metrics: dict, view: MetricView) -> str:
    table = [("Benchmark count", metrics[view.key("benchmark_count")])]
    for variant in VARIANT_ORDER:
        key = view.key(f"total_{variant}_loc")
        if key in metrics:
            table.append((f"Total {variant}", metrics[key]))
    for stat, label in (("mean", "Mean per-benchmark"),
                        ("total", "Pooled-total")):
        for subject, reference in COMPARISONS:
            key = view.key(f"{stat}_{subject}_vs_{reference}_pct")
            if key in metrics:
                table.append(
                    (f"{label} {subject} reduction vs {reference}",
                     f"{metrics[key]:.1f}%"))
    return f"### {title}\n\n" + md_table(("Metric", "Value"), table)


def agreement_line(rows: list[dict]) -> str:
    """Whether scc's SLOC and the built-in count line up, and where not."""
    off = [row for row in rows if row["scc_code"] != row["logical_loc"]]
    if not off:
        return f"scc SLOC matches logical LOC on all {len(rows)} files."
    detail = ", ".join(f"{row['benchmark']}/{row['variant']} "
                       f"({row['logical_loc']} vs {row['scc_code']})"
                       for row in off)
    return (f"scc SLOC and logical LOC disagree on {len(off)} of {len(rows)} "
            f"files: {detail}.")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


REPORT_HEADER = """\
# Focused LOC Comparison

Counts come from curated barebones reference files under `loc-analysis/refs`.
Logical LOC excludes blank lines and comments; each excerpt keeps only the
model-specific data movement and compute."""

SCC_NOTE = """\
The scc columns come from `{version}` (<https://github.com/boyter/scc>), run
over the same files with `--count-as ref:<language>`, `--uloc`, and
`--cognitive`. scc is a second opinion on the line count, and carries metrics a
line count misses: cyclomatic complexity, nesting-weighted cognitive
complexity, and ULOC, the count of *distinct* lines -- which collapses the
copy-pasted transfer and kernel boilerplate the lower-level models repeat."""


def build_report(rows, benchmarks, version) -> str:
    selected = selected_benchmarks(rows, benchmarks)
    legacy = [b for b in selected if b in LEGACY_BENCHMARKS]

    blocks = [REPORT_HEADER, SCC_NOTE.format(version=version),
              agreement_line(rows), "## Overall"]
    for view in VIEWS:
        blocks.append(metric_table(
            f"{view.title}, all {len(selected)} benchmarks",
            aggregate(rows, selected, view), view))
        if legacy and len(legacy) != len(selected):
            blocks.append(metric_table(
                f"{view.title}, original {len(legacy)} benchmarks",
                aggregate(rows, legacy, view), view))

    headings = ["Variant", *(column.heading for column in COLUMNS),
                "Reference file"]
    for benchmark in benchmarks:
        variants = by_variant(rows, benchmark)
        if not variants:
            continue
        table = [[variant, *(variants[variant][column.name]
                             for column in COLUMNS),
                  f"`{variants[variant]['path']}`"]
                 for variant in VARIANT_ORDER if variant in variants]
        blocks.append(f"## {benchmark}\n\n" + md_table(headings, table))

    return "\n\n".join(blocks) + "\n"


def build_stdout(rows, benchmarks) -> str:
    keys = ["benchmark", "variant", *(column.name for column in COLUMNS)]
    blocks = ["\n".join([",".join(keys)] +
                        [",".join(str(row[key]) for key in keys)
                         for row in rows])]

    selected = selected_benchmarks(rows, benchmarks)
    for view in VIEWS:
        metrics = aggregate(rows, selected, view)
        lines = []
        for subject, reference in COMPARISONS:
            key = view.key(f"mean_{subject}_vs_{reference}_pct")
            if key in metrics:
                lines.append(f"overall,{key}={metrics[key]:.1f}")
        blocks.append("\n".join(lines))

    blocks.append(f"# {agreement_line(rows)}")
    return "\n\n".join(blocks)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version = scc_version(args.scc_bin)
    rows = summarize(args.benchmarks, args.variants, args.scc_bin)
    selected = selected_benchmarks(rows, args.benchmarks)

    write_csv(output_dir / "summary.csv", rows,
              ["benchmark", "variant", "path",
               *(column.name for column in COLUMNS)])
    for name, subset in (("overall_metrics.csv", selected),
                         ("overall_metrics_legacy_six.csv",
                          [b for b in selected if b in LEGACY_BENCHMARKS])):
        if not subset:
            continue
        metrics = {}
        for view in VIEWS:
            metrics.update(aggregate(rows, subset, view))
        write_csv(output_dir / name, [metrics], list(metrics.keys()))

    (output_dir / "report.md").write_text(
        build_report(rows, args.benchmarks, version), encoding="utf-8")
    print(build_stdout(rows, args.benchmarks))


if __name__ == "__main__":
    main()
