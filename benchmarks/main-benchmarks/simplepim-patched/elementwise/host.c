#include <assert.h>
#include <benchmark.h>
#include <dpu.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/map/Map.h"
#include "processing/zip/Zip.h"

// Re-uploads data into an already-scattered table, reusing the existing MRAM
// region. simplepim_scatter refuses to overwrite, so we replicate just the
// host-to-DPU transfer step here to keep the table metadata stable.
static void rescatter_to_existing(const char* table_id, void* elements,
                                  simplepim_management_t* mgmt) {
  table_host_t* t = lookup_table(table_id, mgmt);
  uint32_t curr_offset = t->start;
  uint32_t len_per_dpu_in_byte = t->end - t->start;
  struct dpu_set_t set = mgmt->set;
  struct dpu_set_t dpu;
  int i;
  DPU_FOREACH(set, dpu, i) {
    DPU_ASSERT(dpu_prepare_xfer(
        dpu, &((char*)elements)[(uint64_t)i * len_per_dpu_in_byte]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                           curr_offset, len_per_dpu_in_byte, DPU_XFER_DEFAULT));
}

// simplepim_gather() compacts from a padded temporary; uniform shards need no
// such copy.  Read-side counterpart to rescatter_to_existing() above.
static void* gather_direct(const char* table_id, simplepim_management_t* mgmt) {
  table_host_t* t = lookup_table(table_id, mgmt);
  uint32_t num_dpus = mgmt->num_dpus;
  uint32_t type_size = t->table_type_size;
  uint32_t* lens = t->lens_each_dpu;

  uint64_t shard_bytes = (uint64_t)lens[0] * type_size;
  for (uint32_t i = 1; i < num_dpus; i++) {
    if ((uint64_t)lens[i] * type_size != shard_bytes) {
      fprintf(stderr, "gather_direct: ragged shards need the staged path\n");
      exit(1);
    }
  }
  if (shard_bytes % 8 != 0) {
    fprintf(stderr, "gather_direct: shard of %lu bytes is not 8-byte aligned\n",
            (unsigned long)shard_bytes);
    exit(1);
  }

  void* res = malloc((uint64_t)t->len * type_size + 2048);
  if (!res) {
    fprintf(stderr, "gather_direct: out of memory\n");
    exit(1);
  }
  struct dpu_set_t set = mgmt->set;
  struct dpu_set_t dpu;
  int i;
  DPU_FOREACH(set, dpu, i) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, (char*)res + (uint64_t)i * shard_bytes));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
                           t->start, shard_bytes, DPU_XFER_DEFAULT));
  return res;
}

static void free_handle(handle_t* handle) {
  if (!handle) {
    return;
  }
  free(handle->bin_location);
  free(handle->so_bin_location);
  free(handle);
}

void init(T* A) {
  for (uint64_t i = 0; i < nr_elements; i++) {
    A[i] = rand() % 10;
  }
}

void vector_addition_host(T* A, T* B, T* res) {
  omp_set_num_threads(16);
#pragma omp parallel for
  for (uint64_t i = 0; i < nr_elements; i++) {
    res[i] = OPERATION(A[i], B[i]);
  }
}

void run() {
  BenchStages stages;       // steady-loop stages (+ one-time setup)
  BenchStages warm_stages;  // cold warmup-loop stages (the cold-start premium)
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t* table_management = table_management_init(dpu_number);
  bench_stage_end(&stages);
  T* A = NULL;
  T* B = NULL;

  // Inputs only; the reference result is loaded/compared under check below.
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  A = (T*)malloc_scatter_aligned(nr_elements, sizeof(T), table_management);
  B = (T*)malloc_scatter_aligned(nr_elements, sizeof(T), table_management);
  bench_stage_end(&stages);
  if (load_ref) {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    char path[1024];
    printf("Loading reference data from %s...\n", ref_path);
    sprintf(path, "%s/ref_a.bin", ref_path);
    bench_load_bin(path, A, nr_elements * sizeof(T));
    sprintf(path, "%s/ref_b.bin", ref_path);
    bench_load_bin(path, B, nr_elements * sizeof(T));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    init(A);
    init(B);
    bench_stage_end(&stages);
  }

  // create_handle JIT-compiles the DPU kernels at runtime
  // (dpu-upmem-dpurte-clang). That one-time compile is SimplePIM's cold start
  // (PolymerPIM pays its JIT inside the first warmup iteration); time it into
  // warm_stages KERNEL and fold it into the first warmup sample so warmup_ms
  // and kernel_cold_ms agree.
  BenchTimer ch_timer;
  bench_start(&ch_timer, 0);
  bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
  handle_t* add_handle = create_handle("daxby_funcs", MAP);
  handle_t* zip_handle = create_handle("", ZIP);
  bench_stage_end(&warm_stages);
  bench_stop(&ch_timer, 0);
  double create_handle_us = ch_timer.time[0];

  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("t1", A, nr_elements, sizeof(T), table_management);
  simplepim_scatter("t2", B, nr_elements, sizeof(T), table_management);
  table_zip("t1", "t2", "t3", zip_handle, table_management);
  bench_stage_end(&stages);

  T* res = NULL;

  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (int i = 0; i < warmup_iterations; i++) {
    bench_start(&warmup_timer, 0);
    bench_stage_begin(&warm_stages, BENCH_STAGE_WRITE);

    rescatter_to_existing("t1", A, table_management);
    rescatter_to_existing("t2", B, table_management);
    bench_stage_end(&warm_stages);
    bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);

    if (i == 0) {
      handle_t* warm_add_handle = create_handle("daxby_funcs", MAP);
      free_handle(warm_add_handle);
    }

    table_map("t3", "t4", sizeof(T), add_handle, table_management, 0);
    bench_stage_end(&warm_stages);
    bench_stage_begin(&warm_stages, BENCH_STAGE_READ);
    T* tmp = gather_direct("t4", table_management);
    bench_stage_end(&warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0] +
                                          (i == 0 ? create_handle_us : 0.0));
    free(tmp);
  }
  if (warmup_iterations > 0) {
    bench_stats_print("simplepim-patched_warmup", &warmup_stats);
  }

  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;
  for (int i = 0; i < iterations; i++) {
    if (res) {
      free(res);
    }
    bench_start(&timer, 0);
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    rescatter_to_existing("t1", A, table_management);
    rescatter_to_existing("t2", B, table_management);
    bench_stage_end(&stages);
    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    table_map("t3", "t4", sizeof(T), add_handle, table_management, 0);
    bench_stage_end(&stages);
    bench_stage_begin(&stages, BENCH_STAGE_READ);
    res = gather_direct("t4", table_management);
    bench_stage_end(&stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }
  bench_stats_print("simplepim-patched", &stats);
  bench_stages_report("simplepim-patched", &stages);
  bench_stages_report("simplepim-patched_cold", &warm_stages);

  if (check_correctness) {
    T* correct_res = (T*)calloc((uint64_t)nr_elements, sizeof(T));
    if (load_ref) {
      char path[1024];
      sprintf(path, "%s/ref_res.bin", ref_path);
      bench_load_bin(path, correct_res, nr_elements * sizeof(T));
    } else {
      vector_addition_host(A, B, correct_res);
    }

    int32_t is_correct = 1;
    for (uint64_t i = 0; i < nr_elements; i++) {
      if (res[i] != correct_res[i]) {
        is_correct = 0;
        printf("result mismatch at position %lu, got %d, expected %d \n", i,
               res[i], correct_res[i]);
        break;
      }
    }
    if (is_correct) {
      printf("the result is correct \n");
    }
    free(correct_res);
  }

  if (res) {
    free(res);
  }
  free(A);
  free(B);
  free_handle(add_handle);
  free_handle(zip_handle);
  table_management_free(table_management);
}

int main(int argc, char* argv[]) {
  run();
  return 0;
}
