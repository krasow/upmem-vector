#ifndef MAP_TO_VAL_FUNC_H
#define MAP_TO_VAL_FUNC_H

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../Param.h"
#include "processing/gen_red/GenRedArgs.h"

void start_func(gen_red_arguments_t* args) {
  /* reduction: no auxiliary data needed */
}

void map_to_val_func(void* input, void* output, uint32_t* key) {
  *key = 0;
  *(T*)output = *(T*)input;
}

#endif
