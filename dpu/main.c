#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#include <common.h>

__host DPU_LAUNCH_ARGS args;

BARRIER_INIT(my_barrier, NR_TASKLETS);


#include "binary.inl"
#include "unary.inl"

int (*kernels[KERNEL_COUNT])(void) = {
    // Unary
    unary_float_negate,
    unary_float_abs,
    unary_int_negate,
    unary_int_abs,

    // Binary
    binary_float_add,
    binary_float_subtract,
    binary_int_add,
    binary_int_subtract
};

int main(void) {
    // args.kernel indicates which kernel to run
    if (args.kernel < KERNEL_COUNT) {
        return kernels[args.kernel]();
    } else {
        // invalid kernel ID
        return -1;
    }
}