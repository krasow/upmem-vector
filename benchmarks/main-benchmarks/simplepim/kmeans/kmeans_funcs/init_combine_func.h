#ifndef INIT_COMBINE_FUNC_H
#define INIT_COMBINE_FUNC_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../Param.h"

void init_func(uint32_t size, void* ptr) {
  char* casted_value_ptr = (char*)ptr;
  for (int i = 0; i < size; i++) {
    casted_value_ptr[i] = 0;
  }
}

void combine_func(void* p1, void* p2) {
  /* layout: [int32_t count | T coords[dim]] */
  int32_t* times1 = (int32_t*)p1;
  int32_t* times2 = (int32_t*)p2;
  *times1 += *times2;

  T* ptr1 = (T*)((char*)p1 + sizeof(int32_t));
  T* ptr2 = (T*)((char*)p2 + sizeof(int32_t));
  for (int i = 0; i < dim; i++) {
    ptr1[i] += ptr2[i];
  }
}

#endif
