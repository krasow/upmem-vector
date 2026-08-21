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
bool large = true;
const uint32_t dpu_number = 1024;
int iterations = 1;
#define OPERATION(a, b) abs(-((a + b) - a))
#endif
// PARAM DEFAULTS END
#endif
