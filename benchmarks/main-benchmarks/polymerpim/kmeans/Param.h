#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
typedef int32_t T;
#ifndef RED_T
typedef int32_t RED_T;
#endif

const uint32_t DIM = 10;
const uint32_t K = 10;
const uint64_t N = 536870912;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 1;
const uint32_t check_correctness = 0;
const uint32_t load_ref = 1;
const char* ref_path = "../../cpu-verification/kmeans/data";
const uint32_t seed = 1;
#endif
// PARAM DEFAULTS END
#endif
