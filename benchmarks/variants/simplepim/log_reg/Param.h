#pragma once

#if __has_include("Param.generated.h")
#include "Param.generated.h"
#else
// PARAM DEFAULTS BEGIN
#ifndef PARAM_H
#define PARAM_H

#include <stdint.h>
#include <stdlib.h>

uint32_t print_info = 0;
typedef float T;
const uint32_t dpu_number = 5;
const uint32_t dim = 10;
const uint64_t num_elements = 1000 * dpu_number;
const uint32_t iter = 1;
const float lr = 1e-4;
const uint32_t load_ref = 1;
const char* ref_path = "../../cpu-verification/log_reg/data";
const uint32_t check_correctness = 1;

#endif
// PARAM DEFAULTS END
#endif
