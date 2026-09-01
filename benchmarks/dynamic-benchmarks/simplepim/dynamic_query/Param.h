#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>

typedef int32_t T;
#define QUERY_COLUMNS 4
typedef struct {
  T values[QUERY_COLUMNS];
} query_row_t;

const uint64_t nr_elements = 8388608;
const uint32_t dpu_number = 64;
const uint32_t iterations = 50;
const uint32_t warmup_iterations = 0;
const uint32_t check_correctness = 0;
const uint32_t load_ref = 0;
const uint32_t seed = 1;
const uint32_t columns = QUERY_COLUMNS;
const uint32_t batches_per_query = 5;
const uint32_t query_ops = 6;
const char* query_trace = "../../dynamic_query.csv";

#endif
// PARAM DEFAULTS END
#endif
