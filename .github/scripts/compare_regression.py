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


def load_runs(path):
    runs = {}
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["variant"] not in VARIANTS:
                continue
            if row["status"] != "complete":
                raise ValueError(
                    f"{path}: {row['benchmark']}/{row['variant']} is {row['status']}"
                )
            key = tuple(row[field] for field in KEY_FIELDS)
            if key in runs:
                raise ValueError(f"{path}: duplicate result for {'/'.join(key)}")
            runs[key] = float(row["time"])
    return runs


def report(base_path, candidate_path, threshold):
    base = load_runs(base_path)
    candidate = load_runs(candidate_path)
    missing = sorted(set(candidate) - set(base))
    if missing:
        raise ValueError(
            "base results are missing: "
            + ", ".join(f"{key[0]}/{key[1]}" for key in missing)
        )

    lines = [
        "## Benchmark regression",
        "",
        "| Benchmark | Variant | Base (ms) | PR (ms) | Change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    regressions = []
    for key in sorted(candidate):
        before = base[key]
        after = candidate[key]
        change = 100.0 * (after / before - 1.0)
        lines.append(
            f"| {key[0]} | {key[1]} | {before:.3f} | {after:.3f} | {change:+.1f}% |"
        )
        if change > threshold:
            regressions.append((key, change))

    lines.extend(
        ["", f"Failure threshold: more than {threshold:.0f}% slower than the base branch."]
    )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as output:
            output.write(text)

    if regressions:
        for key, change in regressions:
            print(
                f"regression: {key[0]}/{key[1]} is {change:.1f}% slower",
                file=sys.stderr,
            )
        return 1
    return 0


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
