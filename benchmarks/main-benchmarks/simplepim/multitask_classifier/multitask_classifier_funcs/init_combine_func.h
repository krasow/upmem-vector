#ifndef INIT_COMBINE_FUNC_H
#define INIT_COMBINE_FUNC_H

#include <stdint.h>
#include <stdlib.h>

#include "../Param.h"

void init_func(uint32_t size, void* ptr) {
  char* p = (char*)ptr;
  for (uint32_t i = 0; i < size; i++) {
    p[i] = 0;
  }
}

void combine_func(void* dest, void* src) { *(RED_T*)dest += *(RED_T*)src; }

#endif
