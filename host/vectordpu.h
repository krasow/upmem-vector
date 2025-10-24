#ifndef DPU_VECTOR_H
#define DPU_VECTOR_H

#include <cstdint>
#include <vector>

#include <common.h>
#include <dpu.h>

#include "runtime.inl"
#include "allocator.h"

using std::vector;

// ============================
// DPU Vector
// ============================
template <typename T>
class dpu_vector {
public:
    vector<uint32_t> data_;
    vector<uint32_t> sizes_;
    uint32_t size_;

    explicit dpu_vector(uint32_t n);
    ~dpu_vector();

    T* data();
    uint32_t size() const;

    static dpu_vector<T> from_cpu(T val);
    static vector<T> to_cpu(T val);
};

// ============================
// Kernel selectors
// ============================
template <typename T>
struct BinaryKernelSelector;

// float specialization
template <>
struct BinaryKernelSelector<float> {
    static KernelID add() { return KernelID::K_BINARY_FLOAT_ADD; }
    static KernelID sub() { return KernelID::K_BINARY_FLOAT_SUB; }
};

// int specialization
template <>
struct BinaryKernelSelector<int> {
    static KernelID add() { return KernelID::K_BINARY_INT_ADD; }
    static KernelID sub() { return KernelID::K_BINARY_INT_SUB; }
};


template <typename T>
struct UnaryKernelSelector;

// float specialization
template <>
struct UnaryKernelSelector<float> {
    static KernelID negate() { return KernelID::K_UNARY_FLOAT_NEGATE; }
    static KernelID abs()    { return KernelID::K_UNARY_FLOAT_ABS; }
};

// int specialization
template <>
struct UnaryKernelSelector<int> {
    static KernelID negate() { return KernelID::K_UNARY_INT_NEGATE; }
    static KernelID abs()    { return KernelID::K_UNARY_INT_ABS; }
};


// ============================
// DPU Launch helpers
// ============================
template <typename T>
dpu_vector<T> launch_binop(const dpu_vector<T>& lhs,
                            const dpu_vector<T>& rhs,
                            KernelID kernel_id);

template <typename T>
dpu_vector<T> launch_unary(const dpu_vector<T>& a,
                            KernelID kernel_id);

// ============================
// Operators
// ============================
template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs,
                        const dpu_vector<T>& rhs);

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs,
                        const dpu_vector<T>& rhs);

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a);

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a);

#endif // DPU_VECTOR_H
