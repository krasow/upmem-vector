#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>

typedef int32_t T;
const uint64_t N = 16777216;
const uint32_t iterations = 16;
const uint32_t warmup_iterations = 0;
const uint32_t check_correctness = 0;
const uint32_t load_ref = 0;
const uint32_t seed = 1;
const T tolerance = 48;

#endif
// PARAM DEFAULTS END
#endif
