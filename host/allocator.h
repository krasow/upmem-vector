#ifndef ALLOACTOR_H
#define ALLOACTOR_H 

#include <cstdint>
#include <vector>
using std::vector;


typedef std::pair<vector<uint32_t>, vector<uint32_t>> vector_desc;

class allocator {
    public:
        allocator() {}
        static vector_desc allocator::allocate_upmem_vector(std::size_t n, std::size_t num_dpus);
        static void deallocate_upmem_vector(vector_desc & data, std::size_t num_dpus);
    private:
        std::size_t total_size_;
        vector<void*> allocations_;
        vector<void*> free_list_;
};          

#endif