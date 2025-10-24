#include <mram.h>
#define DEFINE_BINARY_KERNEL(TYPE, OP, SYMBOL)                               \
int binary_##TYPE##_##OP(void) {                                             \
    unsigned int tasklet_id = me();                                          \
    uint32_t num_elems = args.num_elements;                                  \
                                                                             \
    TYPE *lhs_ptr = (TYPE *)args.binary.lhs_offset;                                 \
    TYPE *rhs_ptr = (TYPE *)args.binary.rhs_offset;                                 \
    TYPE *res_ptr = (TYPE *)args.binary.res_offset;                                 \
                                                                             \
    /* WRAM working buffers */                                               \
    TYPE lhs_block[BLOCK_SIZE];                                              \
    TYPE rhs_block[BLOCK_SIZE];                                              \
    TYPE res_block[BLOCK_SIZE];                                              \
                                                                             \
    /* Blocked iteration per tasklet */                                      \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;                 \
         block_loc < num_elems;                                              \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                    \
                                                                             \
        uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems) ?       \
                                (num_elems - block_loc) : BLOCK_SIZE;        \
                                                                             \
        /* Copy block from MRAM to WRAM */                                   \
        for (uint32_t i = 0; i < block_elems; i++) {                         \
            lhs_block[i] = lhs_ptr[block_loc + i];                           \
            rhs_block[i] = rhs_ptr[block_loc + i];                           \
        }                                                                    \
                                                                             \
        /* Compute in WRAM */                                                \
        for (uint32_t i = 0; i < block_elems; i++) {                         \
            res_block[i] = lhs_block[i] SYMBOL rhs_block[i];                 \
        }                                                                    \
                                                                             \
        /* Write result back to MRAM */                                      \
        for (uint32_t i = 0; i < block_elems; i++) {                         \
            res_ptr[block_loc + i] = res_block[i];                           \
        }                                                                    \
    }                                                                        \
    return 0;                                                                \
}

DEFINE_BINARY_KERNEL(float, add, +)
DEFINE_BINARY_KERNEL(float, subtract, -)
DEFINE_BINARY_KERNEL(int, add, +)
DEFINE_BINARY_KERNEL(int, subtract, -)