#ifndef USER_H
#define USER_H

#include <stdio.h>
#include <stdlib.h>

#include "Param.h"
#include "processing/map/MapArgs.h"

void start_func(map_arguments_t* args) {}

void map_func(void* input, void* res) {
  *(T*)res = OPERATION(((T*)input)[0], ((T*)input)[1]);
}

#endif
