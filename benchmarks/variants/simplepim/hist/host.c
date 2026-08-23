#include <assert.h>
#include <benchmark.h>
#include <dpu.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Param.h"
#include "communication/CommOps.h"
#include "management/Management.h"
#include "processing/ProcessingHelperHost.h"
#include "processing/gen_red/GenRed.h"

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

void init_data(T* A) {
  for (unsigned long i = 0; i < nr_elements; i++) {
    A[i] = i % 4096;
  }
}

void zero_out_hist(uint32_t* histo) {
  for (int i = 0; i < bins; i++) {
    histo[i] = 0;
  }
}

void histogram_host(uint32_t* histo, T* A) {
  for (unsigned int j = 0; j < nr_elements; j++) {
    T d = A[j];
    histo[(d * bins) >> DEPTH] += 1;
  }
}

void printf_hist(uint32_t* histo) {
  printf("the bins:\n");
  for (int i = 0; i < bins; i++) {
    printf("%u ", histo[i]);
    if (i % 8 == 7) {
      printf("\n");
    }
  }
}

void run() {
  printf("the number of elements %lu\n", nr_elements);
  BenchStages stages;       // steady-loop stages (+ one-time setup)
  BenchStages warm_stages;  // cold warmup-loop stages (the cold-start premium)
  bench_stages_init(&stages);
  bench_stages_init(&warm_stages);

  bench_stage_begin(&stages, BENCH_STAGE_INIT);
  simplepim_management_t* table_management = table_management_init(dpu_number);
  bench_stage_end(&stages);
  T* A = NULL;

  // Inputs only; the reference result is loaded/compared under check below.
  bench_stage_begin(&stages, BENCH_STAGE_ALLOC);
  A = (T*)malloc_scatter_aligned(nr_elements, sizeof(T), table_management);
  bench_stage_end(&stages);
  if (load_ref) {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    char path[1024];
    printf("Loading reference data from %s...\n", ref_path);
    sprintf(path, "%s/ref_t1.bin", ref_path);
    bench_load_bin(path, A, nr_elements * sizeof(T));
    bench_stage_end(&stages);
  } else {
    bench_stage_begin(&stages, BENCH_STAGE_LOAD);
    init_data(A);
    bench_stage_end(&stages);
  }

  bench_stage_begin(&stages, BENCH_STAGE_WRITE);
  simplepim_scatter("t1", A, nr_elements, sizeof(T), table_management);
  bench_stage_end(&stages);
  printf("end of data transfer\n");

  // create_handle JIT-compiles the DPU kernel at runtime
  // (dpu-upmem-dpurte-clang). That one-time compile is SimplePIM's cold start
  // (PolymerPIM pays its JIT inside the first warmup iteration); time it into
  // warm_stages KERNEL and fold it into the first warmup sample so warmup_ms
  // and kernel_cold_ms agree.
  BenchTimer ch_timer;
  bench_start(&ch_timer, 0);
  bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
  handle_t* va_handle = create_handle("hist_funcs", REDUCE);
  bench_stage_end(&warm_stages);
  bench_stop(&ch_timer, 0);
  double create_handle_us = ch_timer.time[0];

  uint32_t* res = NULL;

  // Time the full round trip (host-to-DPU re-upload + reduction kernel +
  // result gather) to match the polymerpim and baseline histogram
  // measurements, which both include the host-to-DPU data transfer.
  BenchTimer warmup_timer;
  BenchStats warmup_stats;
  bench_stats_init(&warmup_stats);
  for (uint32_t i = 0; i < warmup_iterations; i++) {
    bench_start(&warmup_timer, 0);
    bench_stage_begin(&warm_stages, BENCH_STAGE_WRITE);
    rescatter_to_existing("t1", A, table_management);
    bench_stage_end(&warm_stages);
    bench_stage_begin(&warm_stages, BENCH_STAGE_KERNEL);
    uint32_t* tmp = (uint32_t*)table_gen_red("t1", "t2", sizeof(uint32_t), bins,
                                             va_handle, table_management, 0);
    bench_stage_end(&warm_stages);
    bench_stop(&warmup_timer, 0);
    bench_stats_update(&warmup_stats, warmup_timer.time[0] +
                                          (i == 0 ? create_handle_us : 0.0));
    free(tmp);
  }
  if (warmup_iterations > 0)
    bench_stats_print("simplepim_warmup", &warmup_stats);

  BenchStats stats;
  bench_stats_init(&stats);
  BenchTimer timer;
  for (uint32_t i = 0; i < iterations; i++) {
    if (res) free(res);
    bench_start(&timer, 0);
    bench_stage_begin(&stages, BENCH_STAGE_WRITE);
    rescatter_to_existing("t1", A, table_management);
    bench_stage_end(&stages);
    bench_stage_begin(&stages, BENCH_STAGE_KERNEL);
    res = (uint32_t*)table_gen_red("t1", "t2", sizeof(uint32_t), bins,
                                   va_handle, table_management, 0);
    bench_stage_end(&stages);
    bench_stop(&timer, 0);
    bench_stats_update(&stats, timer.time[0]);
  }
  bench_stats_print("simplepim", &stats);
  bench_stages_report("simplepim", &stages);
  bench_stages_report("simplepim_cold", &warm_stages);

  if (print_info) {
    printf("DPU histogram:\n");
    printf_hist(res);
  }

  // check correctness
  if (check_correctness) {
    uint32_t* cpu_histo = (uint32_t*)calloc(bins, sizeof(uint32_t));
    if (load_ref) {
      char path[1024];
      sprintf(path, "%s/ref_res.bin", ref_path);
      bench_load_bin(path, cpu_histo, bins * sizeof(uint32_t));
    } else {
      zero_out_hist(cpu_histo);
      histogram_host(cpu_histo, A);
    }

    if (print_info) {
      printf("CPU histogram:\n");
      printf_hist(cpu_histo);
    }

    int correct = 1;
    for (uint64_t i = 0; i < (uint64_t)bins; i++) {
      if (cpu_histo[i] != res[i]) {
        printf("MISMATCH at bin %lu: cpu=%u dpu=%u\n", i, cpu_histo[i], res[i]);
        correct = 0;
      }
    }
    if (correct) {
      printf("the result is correct\n");
    }
    free(cpu_histo);
  }

  free(res);
}

int main(int argc, char** argv) {
  srand(17);
  run();
  return 0;
}
