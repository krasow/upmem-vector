#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

typedef struct {
  uint32_t lhs_offset;
  uint32_t rhs_offset;
  uint32_t res_offset;
  uint32_t num_elements;
  uint32_t kernel_id;
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;

__host DPU_LAUNCH_ARGS args;

int kernel_1(void) { return 1; }

int kernel_2(void) { return 2; }

int (*kernels[])(void) = {kernel_1, kernel_2};

int main(void) { return kernels[args.kernel_id](); }
