ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

NR_TASKLETS ?= 12

BIN_DIR := bin
__dirs := $(shell mkdir -p $(BIN_DIR))

HOST := $(BIN_DIR)/host_baseline
DPU := $(BIN_DIR)/baseline.dpu

all : $(HOST) $(DPU)

# Sources
# HOST_SRC :=
# DPU_SRC :=


# Compiler flags
CFLAGS := -O3 -fopenmp
LIBS := -lm -ldl `dpu-pkg-config --cflags --libs dpu`
DPU_FLAGS := -Wall -Wextra -O3 -DNR_TASKLETS=$(NR_TASKLETS)

$(BIN_DIR)/host_slice: host/slice.cc
	gcc $(CFLAGS) host/slice.cc -o $(BIN_DIR)/host_slice $(LIBS)

$(BIN_DIR)/host_baseline: host/baseline.cc
	gcc $(CFLAGS) host/baseline.cc -o $(BIN_DIR)/host_baseline $(LIBS)

$(BIN_DIR)/baseline.dpu: dpu/dpu_baseline.c
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $(BIN_DIR)/baseline.dpu dpu/dpu_baseline.c

$(BIN_DIR)/slice.dpu: dpu/dpu_slice.c
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $(BIN_DIR)/slice.dpu dpu/dpu_slice.c

clean:
	rm -rf $(BIN_DIR)
