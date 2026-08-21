#include <stdio.h>
#include <stdlib.h>
#ifndef DPURT
#define DPURT
#include <dpu>  // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include "../Param.h"
#include "timer.h"

typedef struct {
  uint32_t lhs_offset;
  uint32_t rhs_offset;
  uint32_t res_offset;
  uint32_t num_elements;
  uint32_t kernel_id;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

int main() {
  int nr_of_dpus = dpu_number;
  dpu_set_t dpu_set;

  CHECK_UPMEM(dpu_alloc(nr_of_dpus, "backend=hw", &dpu_set));

  DPU_LAUNCH_ARGS args[nr_of_dpus];

  for (uint32_t i = 0; i < nr_of_dpus; i++) {
    args[i].num_elements = 0;
    args[i].lhs_offset = 0;
    args[i].rhs_offset = 0;
    args[i].res_offset = 0;
    args[i].kernel_id = 1;
  }

  Timer timer;
  start(&timer, 0, 0);

  const char *dpu_bin = large ? "./bin/large.dpu" : "./bin/small.dpu";
  for (int i = 0; i < iterations; i++) {
    CHECK_UPMEM(dpu_load(dpu_set, dpu_bin, nullptr));
    // for (uint32_t i = 0; i < nr_of_dpus; i++) {
    // 	args[i].num_elements = 0;
    // 	args[i].lhs_offset = 0;
    // 	args[i].rhs_offset = 0;
    // 	args[i].res_offset = 0;
    // 	args[i].kernel_id = 0;
    // }

    // dpu_set_t dpu;
    // uint32_t idx_dpu = 0;
    // DPU_FOREACH(dpu_set, dpu, idx_dpu) {
    // 	CHECK_UPMEM(dpu_prepare_xfer(dpu, &args[idx_dpu]));
    // }
    // CHECK_UPMEM(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "args", 0,
    // 			sizeof(args[0]), DPU_XFER_DEFAULT));

    CHECK_UPMEM(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
  }

  stop(&timer, 0);

  printf("baseline (ms): ");
  print(&timer, 0, 1);
  printf("\n");

  CHECK_UPMEM(dpu_free(dpu_set));

  return 0;
}
