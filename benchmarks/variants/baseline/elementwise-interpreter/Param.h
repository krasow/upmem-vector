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
const uint32_t check_correctness = false;
const uint32_t dpu_number = 1024;
uint32_t print_info = 0;
uint64_t nr_elements = 3221225472;
int iterations = 1;
const int warmup_iterations = 1;
#endif
// PARAM DEFAULTS END
#endif
