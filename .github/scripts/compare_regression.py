#!/usr/bin/env python3

import argparse
import csv
import os
import sys


VARIANTS = {"polymerpim", "julia"}
KEY_FIELDS = (
    "benchmark",
    "variant",
    "dpus",
    "elements_per_dpu",
    "iterations",
    "trial",
)


COUNTER = "jit_kernel_compiles"


def load_runs(path):
    runs = {}
    counters = {}
    failed = set()
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["variant"] not in VARIANTS:
                continue
            key = tuple(row[field] for field in KEY_FIELDS)
            if row["status"] != "complete" or not row["time"]:
                failed.add(key)
                continue
            runs[key] = float(row["time"])
            if row.get(COUNTER):
                counters[key] = float(row[COUNTER])
    return runs, counters, failed - set(runs)


def report(base_path, candidate_path, threshold):
    base, base_counters, _ = load_runs(base_path)
    candidate, candidate_counters, candidate_failed = load_runs(candidate_path)
    # Deterministic, so any increase is a real change -- no threshold needed.
    recompiles = [
        (key, base_counters[key], candidate_counters[key])
        for key in sorted(candidate_counters)
        if key in base_counters and candidate_counters[key] > base_counters[key]
    ]
    # Not fatal: a PR that fixes a crash on base would always fail here.
    missing = sorted(set(candidate) - set(base))

    lines = [
        "## Benchmark regression",
        "",
        "| Benchmark | Variant | Base (ms) | PR (ms) | Change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    regressions = []
    for key in sorted(candidate):
        if key not in base:
            continue
        before = base[key]
        after = candidate[key]
        change = 100.0 * (after / before - 1.0)
        lines.append(
            f"| {key[0]} | {key[1]} | {before:.3f} | {after:.3f} | {change:+.1f}% |"
        )
        if change > threshold:
            regressions.append((key, change))

    if recompiles:
        lines.extend(["", "**More JIT kernels compiled than base:**"])
        lines.extend(f"- {k[0]}/{k[1]}: {int(b)} -> {int(c)}"
                     for k, b, c in recompiles)
    for label, keys in (("Failed on this PR", candidate_failed),
                       ("Not compared (no base result)", missing)):
        if keys:
            lines.extend(["", f"{label}:"])
            lines.extend(f"- {key[0]}/{key[1]}" for key in sorted(keys))
    lines.extend(
        ["", f"Failure threshold: more than {threshold:.0f}% slower than the base branch."]
    )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as output:
            output.write(text)

    for key in sorted(candidate_failed):
        print(f"failed on this PR: {key[0]}/{key[1]}", file=sys.stderr)
    for key, change in regressions:
        print(f"regression: {key[0]}/{key[1]} is {change:.1f}% slower", file=sys.stderr)
    for key, b, c in recompiles:
        print(f"more JIT compiles: {key[0]}/{key[1]} {int(b)} -> {int(c)}",
              file=sys.stderr)
    return 1 if (regressions or candidate_failed or recompiles) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("candidate")
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()
    try:
        return report(args.base, args.candidate, args.threshold)
    except (OSError, KeyError, ValueError) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
