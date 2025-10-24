#include "vectordpu.h"

#include <cassert>
#include <cstdio>

// ============================
// DPU Vector
// ============================
template <typename T>
dpu_vector<T>::dpu_vector(uint32_t n)
    : size_(n)
{
    auto& runtime = DpuRuntime::get();
    data_ = runtime.get_allocator().allocate_upmem_vector(n * sizeof(T), runtime.num_dpus());
}

template <typename T>
dpu_vector<T>::~dpu_vector()
{
    auto& runtime = DpuRuntime::get();
    runtime.get_allocator().deallocate_upmem_vector(data_, sizes_);
}

template <typename T>
T* dpu_vector<T>::data()
{
    return reinterpret_cast<T*>(data_.data());
}

template <typename T>
uint32_t dpu_vector<T>::size() const
{
    return size_;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(T val)
{
    dpu_vector<T> vec(1);
    // TODO: implement transfer to DPU memory
    return vec;
}

template <typename T>
vector<T> dpu_vector<T>::to_cpu(T val)
{
    vector<T> res;
    // TODO: implement transfer from DPU memory
    return res;
}

template <typename T>
dpu_vector<T> launch_binop(const dpu_vector<T>& lhs,
                            const dpu_vector<T>& rhs,
                            KernelID kernel_id)
{
    assert(lhs.size() == rhs.size());
    dpu_vector<T> res(lhs.size());

    auto& runtime = DpuRuntime::get();
    uint32_t nr_of_dpus = runtime.num_dpus();
    DPU_LAUNCH_ARGS args[nr_of_dpus];

    for (uint32_t i = 0; i < nr_of_dpus; i++) {
        args[i].kernel = static_cast<uint32_t>(kernel_id);
        args[i].is_binary = 1;
        args[i].num_elements = lhs.size();
        args[i].size_type = sizeof(T);
        args[i].binary.lhs_offset = reinterpret_cast<uint32_t>(lhs.data()[i]);
        args[i].binary.rhs_offset = reinterpret_cast<uint32_t>(rhs.data()[i]);
        args[i].binary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
    }

    dpu_set_t& dpu_set = runtime.dpu_set();
    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &args[idx_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             "DPU_LAUNCH_ARGS", 0, sizeof(args[0]),
                             DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    return res;
}

template <typename T>
dpu_vector<T> launch_unary(const dpu_vector<T>& a,
                            KernelID kernel_id)
{
    dpu_vector<T> res(a.size());

    auto& runtime = DpuRuntime::get();
    uint32_t nr_of_dpus = runtime.num_dpus();
    DPU_LAUNCH_ARGS args[nr_of_dpus];

    for (uint32_t i = 0; i < nr_of_dpus; i++) {
        args[i].kernel = static_cast<uint32_t>(kernel_id);
        args[i].is_binary = 0;
        args[i].num_elements = a.size();
        args[i].size_type = sizeof(T);
        args[i].unary.lhs_offset = reinterpret_cast<uint32_t>(a.data()[i]);
        args[i].unary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
    }

    dpu_set_t& dpu_set = runtime.dpu_set();
    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &args[idx_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             "DPU_LAUNCH_ARGS", 0, sizeof(args[0]),
                             DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    return res;
}

// Binary operators
template <typename T>
dpu_vector<T> operator+(const dpu_vector<T>& lhs,
                        const dpu_vector<T>& rhs)
{
    return launch_binop(lhs, rhs, BinaryKernelSelector<T>::add());
}

template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& lhs,
                        const dpu_vector<T>& rhs)
{
    return launch_binop(lhs, rhs, BinaryKernelSelector<T>::sub());
}

// Unary operators
template <typename T>
dpu_vector<T> operator-(const dpu_vector<T>& a)
{
    return launch_unary(a, UnaryKernelSelector<T>::negate());
}

template <typename T>
dpu_vector<T> abs(const dpu_vector<T>& a)
{
    return launch_unary(a, UnaryKernelSelector<T>::abs());
}
