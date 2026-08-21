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
typedef int32_t T;
const uint32_t dpu_number = 2048;
const uint32_t k = 10;
const uint32_t dim = 10;
const uint64_t num_elements = 536870912;
const uint32_t iter = 1;
const uint32_t warmup_iterations = 1;
const uint32_t load_ref = 1;
const char* ref_path = "../../cpu-verification/kmeans/data";
const uint32_t check_correctness = 0;

#endif
// PARAM DEFAULTS END
#endif
