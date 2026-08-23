#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H

#include <stdint.h>
#include <stdlib.h>

const uint32_t print_info = 1;
typedef int T;

const uint32_t check_correctness = 0;
const uint32_t load_ref = 1;
const char* ref_path = "../../cpu-verification/linreg/data";
const uint32_t seed = 1;
#ifndef RED_T
typedef int32_t RED_T;
#endif

const uint32_t dpu_number = 2048;
const uint32_t dim = 10;
const uint64_t nr_elements = 1073741824;
const uint32_t iterations = 1;
const uint32_t warmup_iterations = 1;
const float lr = 1e-4;
const uint32_t shift_amount = 0;
const uint32_t prevent_overflow_shift_amount = 12;

#endif
// PARAM DEFAULTS END
#endif
