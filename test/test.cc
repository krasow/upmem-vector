/* This test will use the library to perform element wise tests on DPU data.

   The main comparison is between our driver implementation and the default UPMEM driver.
   UPMEM driver does a transpose of data to distribute it across DPUs, while our driver returns an mmaped region.
   With our driver, it doesn't transpose data and places data in a round robin fashion across DPUs. It is expected that 
   high level element wise operations will work correctly and faster on our implementation due to the lack of transpose.


   David Krasowska, October 2025
*/


#include <vectordpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

using test_error = uint32_t;

#define TEST_UNIMPLIMENTED 2
#define TEST_ERROR 1
#define TEST_SUCCESS 0


test_error compare_cpu(vector<int>& a, vector<int>& b, dpu_vector<int>& res) {
    vector<int> cpu_res = res.to_cpu();
    for (uint32_t i = 0; i < res.size(); i++) {
        if (cpu_res[i] == a[i] + b[i]) {
            continue;
        } else {
            return TEST_ERROR;
        }
    }
    return TEST_SUCCESS;
}

test_error our_driver_test() {
    return TEST_UNIMPLIMENTED;
}

test_error upmem_driver_test() {
    const uint32_t N = 1024 * 1024; // 1M elements
    vector<int> a(N);           
    vector<int> b(N); 
    for (uint32_t i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
    dpu_vector<int> da = dpu_vector<int>::from_cpu(a);
    dpu_vector<int> db = dpu_vector<int>::from_cpu(b);
    dpu_vector<int> res = da + db;

    return compare_cpu(a, b, res);
}


int main(void) {
    assert(upmem_driver_test() == TEST_SUCCESS);
    return 0;
}