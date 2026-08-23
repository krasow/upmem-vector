#ifndef INIT_COMBINE_FUNC_H
#define INIT_COMBINE_FUNC_H

#include "../Param.h"

void init_func(uint32_t size, void *ptr) {
  (void)size;
  vector_search_result_init((vector_search_result_t *)ptr);
}

void combine_func(void *dest, void *src) {
  vector_search_result_merge((vector_search_result_t *)dest,
                             (const vector_search_result_t *)src);
}

#endif
