#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>

typedef int32_t T;
#ifndef RED_T
typedef uint32_t RED_T;
#endif

#define DEPTH 12
#define BINS 256

const uint64_t N = 2147483648;
const uint32_t dpu_number = 2048;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 1;
const uint32_t check_correctness = 0;
const uint32_t load_ref = 1;
const char* ref_path = "../../cpu-verification/hist/data";
const uint32_t seed = 1;

#endif
// PARAM DEFAULTS END
#endif
