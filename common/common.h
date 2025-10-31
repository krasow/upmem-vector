#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#define DPU_RUNTIME "/home/david/upmem-driver-test/library/bin/runtime.dpu"

#define BLOCK_SIZE_LOG2 5              // e.g., 32 elements per block
#define BLOCK_SIZE (1U << BLOCK_SIZE_LOG2)

typedef enum {
    // Unary
    K_UNARY_FLOAT_NEGATE,
    K_UNARY_FLOAT_ABS,
    K_UNARY_INT_NEGATE,
    K_UNARY_INT_ABS,

    // Binary
    K_BINARY_FLOAT_ADD,
    K_BINARY_FLOAT_SUB,
    K_BINARY_INT_ADD,
    K_BINARY_INT_SUB,

    KERNEL_COUNT
} KernelID;

typedef struct {
    uint32_t kernel;       // 4
    uint32_t num_elements; // 4
    uint32_t size_type;    // 4

    union {
        struct {           // binary ops
            uint32_t lhs_offset;
            uint32_t rhs_offset;
            uint32_t res_offset;
        } binary;          // 12
        struct {           // unary ops
            uint32_t rhs_offset;
            uint32_t res_offset;
            uint32_t pad;   // pad unary to 12 bytes
        } unary;
    };

    uint8_t is_binary;     // 1
    uint8_t pad[7];        // pad struct to 32 bytes
} __attribute__((aligned(8))) DPU_LAUNCH_ARGS;


#endif // COMMON_H