#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>

typedef int32_t T;
const uint64_t N = 2147483648;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 1;
const uint32_t seed = 1;

#endif
// PARAM DEFAULTS END
#endif
