#include <mram.h>

#define NEGATE(x) (-(x))
#define ABS(x) ((x) < 0 ? -(x) : (x))

#define DEFINE_UNARY_KERNEL(TYPE, OP, FUNC)                               \
int unary_##TYPE##_##OP(void) {                                                \
    unsigned int tasklet_id = me();                                            \
    uint32_t num_elems = args.num_elements;                                    \
                                                                               \
    TYPE *rhs_ptr = (TYPE *)args.unary.rhs_offset;                             \
    TYPE *res_ptr = (TYPE *)args.unary.res_offset;                             \
                                                                               \
    /* WRAM working buffer */                                                  \
    TYPE lhs_block[BLOCK_SIZE];                                                \
    TYPE res_block[BLOCK_SIZE];                                                \
                                                                               \
    /* Blocked iteration per tasklet */                                        \
    for (uint32_t block_loc = tasklet_id << BLOCK_SIZE_LOG2;                   \
         block_loc < num_elems;                                                \
         block_loc += (NR_TASKLETS << BLOCK_SIZE_LOG2)) {                      \
                                                                               \
        uint32_t block_elems = (block_loc + BLOCK_SIZE >= num_elems) ?         \
                                (num_elems - block_loc) : BLOCK_SIZE;          \
                                                                               \
        /* Copy block from MRAM to WRAM */                                     \
        for (uint32_t i = 0; i < block_elems; i++) {                           \
            lhs_block[i] = rhs_ptr[block_loc + i];                             \
        }                                                                      \
                                                                               \
        /* Compute in WRAM */                                                  \
        for (uint32_t i = 0; i < block_elems; i++) {                           \
            res_block[i] = FUNC(lhs_block[i]);                                 \
        }                                                                      \
                                                                               \
        /* Write result back to MRAM */                                        \
        for (uint32_t i = 0; i < block_elems; i++) {                           \
            res_ptr[block_loc + i] = res_block[i];                             \
        }                                                                      \
    }                                                                          \
    return 0;                                                                  \
}


DEFINE_UNARY_KERNEL(float, negate, NEGATE)
DEFINE_UNARY_KERNEL(int,   negate, NEGATE)
DEFINE_UNARY_KERNEL(float, abs, ABS)
DEFINE_UNARY_KERNEL(int,   abs, ABS)
