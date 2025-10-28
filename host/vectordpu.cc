#include "vectordpu.h"
#include "runtime.h"
#include "logger.inl"

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
    
    if(runtime.is_initialized() == false) {
        // throw std::runtime_error("DPU runtime not initialized!");
        runtime.init(16);
    }

    data_ = runtime.get_allocator().allocate_upmem_vector(n, sizeof(T));
}

template <typename T>
dpu_vector<T>::~dpu_vector()
{
    auto& runtime = DpuRuntime::get();
    runtime.get_allocator().deallocate_upmem_vector(data_);
}

template <typename T>
vector<uint32_t> dpu_vector<T>::data() const
{
    // data_ is vector_desc std::pair<vector<uint32_t>, vector<uint32_t>>
    // where first element is vector of pointers to DPU memory per DPU
    // and second element is vector of sizes per DPU
    return data_.first;
}

template <typename T>
uint32_t dpu_vector<T>::size() const
{
    return size_;
}

template <typename T>
dpu_vector<T> dpu_vector<T>::from_cpu(vector<T>& cpu_data)
{
    dpu_vector<T> vec(cpu_data.size());
    // TODO: implement transfer to DPU memory
    // .data returns a std::pair<vector<uint32_t>, vector<uint32_t>>
    // the first element is vector of pointers to DPU memory per DPU
    // the second element is vector of sizes per DPU
    auto desc = vec.data_desc();

    #if ENABLE_DPU_LOGGING >= 2
        print_vector_desc(desc);
    #endif
    
    auto& runtime = DpuRuntime::get();

    dpu_set_t& dpu_set = runtime.dpu_set();
    dpu_set_t dpu;
    uint32_t idx_dpu = 0;

    char *cpu_vec = (char*)cpu_data.data();
    size_t element = 0;

    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
		DPU_ASSERT(dpu_prepare_xfer(dpu, &(cpu_vec[element])));
        element += desc.second[idx_dpu];
	}

    uint32_t mram_location = desc.first[0];
    size_t xfer_size = desc.second[0];

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 
                            mram_location, xfer_size, DPU_XFER_DEFAULT));

    #if ENABLE_DPU_LOGGING == 1
        std::cout << "[debug-help] Transferred " << cpu_data.size() << " elements to DPU." << std::endl;
    #endif 
    return vec;
}

template <typename T>
vector<T> dpu_vector<T>::to_cpu()
{
    auto desc = this->data_desc();  // pair< vector<uint32_t>, vector<uint32_t> >
    
    #if ENABLE_DPU_LOGGING >= 2
        print_vector_desc(desc);
    #endif

    auto& runtime = DpuRuntime::get();
    dpu_set_t& dpu_set = runtime.dpu_set();
    dpu_set_t dpu;

    // Allocate CPU buffer large enough to hold all data
    vector<T> cpu_data(this->size());
    char* cpu_vec = reinterpret_cast<char*>(cpu_data.data());

    uint32_t idx_dpu = 0;
    size_t element = 0;

    // Prepare DMA transfers from each DPU
    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &(cpu_vec[element])));
        element += desc.second[idx_dpu];
    }

    uint32_t mram_location = desc.first[0];
    size_t xfer_size = desc.second[0];

    // Perform the actual transfer from DPU MRAM to host memory
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                             mram_location, xfer_size, DPU_XFER_DEFAULT));

    #if ENABLE_DPU_LOGGING == 1
        std::cout << "[debug-help] Retrieved " << cpu_data.size() << " elements from DPU." << std::endl;
    #endif
    return cpu_data;
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
        args[i].is_binary = true;
        args[i].num_elements = lhs.size();
        args[i].size_type = sizeof(T);
        args[i].binary.lhs_offset = reinterpret_cast<uint32_t>(lhs.data()[i]);
        args[i].binary.rhs_offset = reinterpret_cast<uint32_t>(rhs.data()[i]);
        args[i].binary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
    }

    #ifdef ENABLE_DPU_LOGGING
        log_dpu_launch_args(args, nr_of_dpus);
    #endif

    dpu_set_t& dpu_set = runtime.dpu_set();
    dpu_set_t dpu;
    uint32_t idx_dpu = 0;

    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &args[idx_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             "args", 0, sizeof(args[0]),
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
        args[i].is_binary = false;
        args[i].num_elements = a.size();
        args[i].size_type = sizeof(T);
        args[i].unary.rhs_offset = reinterpret_cast<uint32_t>(a.data()[i]);
        args[i].unary.res_offset = reinterpret_cast<uint32_t>(res.data()[i]);
    }

    #ifdef ENABLE_DPU_LOGGING
        log_dpu_launch_args(args, nr_of_dpus);
    #endif

    dpu_set_t& dpu_set = runtime.dpu_set();
    dpu_set_t dpu;
    uint32_t idx_dpu = 0;

    DPU_FOREACH(dpu_set, dpu, idx_dpu) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &args[idx_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             "args", 0, sizeof(args[0]),
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

// Explicit instantiation
template dpu_vector<int> operator+<int>(const dpu_vector<int>&, const dpu_vector<int>&);
template dpu_vector<int> operator-<int>(const dpu_vector<int>&, const dpu_vector<int>&);

template dpu_vector<float> operator+<float>(const dpu_vector<float>&, const dpu_vector<float>&);
template dpu_vector<float> operator-<float>(const dpu_vector<float>&, const dpu_vector<float>&);

template dpu_vector<int> operator-<int>(const dpu_vector<int>&);
template dpu_vector<int> abs<int>(const dpu_vector<int>&);

template dpu_vector<float> operator-<float>(const dpu_vector<float>&);
template dpu_vector<float> abs<float>(const dpu_vector<float>&);

template dpu_vector<int> dpu_vector<int>::from_cpu(std::vector<int>&);
template std::vector<int> dpu_vector<int>::to_cpu();

template dpu_vector<float> dpu_vector<float>::from_cpu(std::vector<float>&);
template std::vector<float> dpu_vector<float>::to_cpu();