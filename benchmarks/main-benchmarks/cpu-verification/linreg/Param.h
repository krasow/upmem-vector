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
const uint64_t N = 1073741824;
const uint32_t DIM = 10;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 1;
const uint32_t check_correctness = 0;
const uint32_t load_ref = 1;
const char* ref_path = "./data";
const uint32_t seed = 1;
const uint32_t scaling_shift = 12;

#endif
// PARAM DEFAULTS END
#endif
