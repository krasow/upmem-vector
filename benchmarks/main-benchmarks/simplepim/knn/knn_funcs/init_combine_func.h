#ifndef INIT_COMBINE_FUNC_H
#define INIT_COMBINE_FUNC_H

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#include "../Param.h"

void init_func(uint32_t size, void* ptr) { *(RED_T*)ptr = INT32_MAX; }

void combine_func(void* dest, void* src) {
  RED_T a = *(RED_T*)dest;
  RED_T b = *(RED_T*)src;
  if (b < a) *(RED_T*)dest = b;
}

#endif
