#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H
#include <stdint.h>
#include <stdlib.h>

typedef uint32_t T;

const uint32_t print_info = 0;
const uint32_t dpu_number = 2048;
uint64_t nr_elements = 2147483648;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 1;
const uint32_t load_ref = 1;
const char* ref_path = "../../cpu-verification/red/data";
const uint32_t check_correctness = 0;

#endif
// PARAM DEFAULTS END
#endif
