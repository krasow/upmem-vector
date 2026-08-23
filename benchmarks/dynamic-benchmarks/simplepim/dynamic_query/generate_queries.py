#!/usr/bin/env python3

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACE = ROOT.parent.parent / "dynamic_query.csv"
PARAM = ROOT / ("Param.generated.h" if (ROOT / "Param.generated.h").is_file()
                else "Param.h")


def configured_ops():
    match = re.search(r"\bquery_ops\s*=\s*(\d+)", PARAM.read_text())
    if not match:
        raise SystemExit(f"query_ops is missing from {PARAM}")
    return int(match.group(1))


def queries():
    rows = []
    for line in TRACE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(line.split(","))
    return rows


def expression(tokens):
    lines = ["  T value = row->values[p % QUERY_COLUMNS];"]
    for token in tokens:
        op, column = token[0], int(token[1:])
        operand = f"row->values[(p + {column}) % QUERY_COLUMNS]"
        if op == "A":
            lines.append(f"  value += {operand};")
        elif op == "D":
            lines.append(f"  value = query_abs(value - {operand});")
        elif op == "V":
            lines.append(f"  value = (query_abs(value) + {operand}) >> 1;")
        elif op == "C":
            lines.append(f"  value = query_abs(value + {operand} - 251);")
        else:
            raise SystemExit(f"unknown operation {token}")
    lines.extend([
        "  T lhs = row->values[p % QUERY_COLUMNS];",
        "  T rhs = row->values[(p + 1) % QUERY_COLUMNS];",
        "  *(T *)output = lhs < rhs ? value : 0;",
    ])
    return "\n".join(lines)


def write_query(index, tokens):
    directory = ROOT / f"query_{index:03d}_funcs"
    directory.mkdir(exist_ok=True)
    map_header = f"""#ifndef DYNAMIC_QUERY_MAP_TO_VAL_H
#define DYNAMIC_QUERY_MAP_TO_VAL_H
#include <defs.h>
#include "../Param.h"
#include "processing/gen_red/GenRedArgs.h"

static uint32_t query_projection[NR_TASKLETS];
static inline T query_abs(T value) {{ return value < 0 ? -value : value; }}
void start_func(gen_red_arguments_t *args) {{
  query_projection[me()] = args->info;
}}
void map_to_val_func(void *input, void *output, uint32_t *key) {{
  query_row_t *row = (query_row_t *)input;
  uint32_t p = query_projection[me()];
  *key = 0;
{expression(tokens)}
}}
#endif
"""
    reduce_header = """#ifndef DYNAMIC_QUERY_REDUCE_H
#define DYNAMIC_QUERY_REDUCE_H
#include <limits.h>
#include "../Param.h"
void init_func(uint32_t size, void *ptr) {
  (void)size;
  *(T *)ptr = INT32_MIN;
}
void combine_func(void *dest, void *src) {
  if (*(T *)src > *(T *)dest) *(T *)dest = *(T *)src;
}
#endif
"""
    (directory / "map_to_val_func.h").write_text(map_header)
    (directory / "init_combine_func.h").write_text(reduce_header)


def main():
    count = configured_ops()
    rows = queries()
    for index, tokens in enumerate(rows):
        if len(tokens) < count:
            raise SystemExit(f"query {index} has fewer than {count} operations")
        write_query(index, tokens[:count])
    print(f"Generated {len(rows)} SimplePIM queries with {count} operations")


if __name__ == "__main__":
    main()
