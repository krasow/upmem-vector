#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>

typedef float T;
const uint32_t dpu_number = 5;
const uint32_t dim = 10;
const uint64_t N = 1000 * 5;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 0;
const uint32_t seed = 42;

#endif
// PARAM DEFAULTS END
#endif
