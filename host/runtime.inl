#include "allocator.h"
class DpuRuntime {
private:
    DpuRuntime() : initialized_(false), allocator_(nullptr) {}

    bool initialized_;
    dpu_set_t dpu_set_;
    uint32_t num_dpus_;
    allocator* allocator_;

public:
    // Delete copy/move
    DpuRuntime(const DpuRuntime&) = delete;
    DpuRuntime& operator=(const DpuRuntime&) = delete;

    // Get singleton instance
    static DpuRuntime& get() {
        static DpuRuntime instance;
        return instance;
    }

    void init(uint32_t num_dpus) {
        if (!initialized_) {
            num_dpus_ = num_dpus;

            // Allocate or acquire DPUs
            DPU_ASSERT(dpu_alloc(num_dpus_, "backend=simulator", &dpu_set_));
            DPU_ASSERT(dpu_load(dpu_set_, DPU_RUNTIUME, NULL));

            printf("DPU runtime initialized with %u DPUs\n", num_dpus_);

            initialized_ = true;
            allocator_ = new allocator(0, 64 * 1024 * 1024, num_dpus_); 
        }
    }

    ~DpuRuntime() {
        delete allocator_;
    }

    dpu_set_t& dpu_set() { return dpu_set_; }
    uint32_t num_dpus() const { return num_dpus_; }
    uint32_t num_tasklets() const { return NR_TASKLETS; }
    allocator& get_allocator() { return *allocator_; }
};
