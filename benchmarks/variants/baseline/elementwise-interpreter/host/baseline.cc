#include "../Param.h"

const uint32_t load_ref = 0;
const char *ref_path = "";
const uint32_t seed = 1;
#define OPERATION(a, b) abs((-((a) + (b))) - (a))
#define ELEMENTWISE_PARAM_INCLUDED 1

#include "../../elementwise/host/baseline.cc"
