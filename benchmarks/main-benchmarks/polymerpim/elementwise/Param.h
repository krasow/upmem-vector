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
const uint32_t check_correctness = 0;
const uint32_t load_ref = 1;
const char* ref_path = "../../cpu-verification/elementwise/data";
const uint32_t seed = 1;

const uint64_t N = 4294967296;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 1;
#define OPERATION(a, b) -abs(((a) + (b)) - (a))
#endif
// PARAM DEFAULTS END
#endif
