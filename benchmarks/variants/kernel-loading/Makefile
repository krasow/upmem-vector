ifndef UPMEM_HOME
$(error UPMEM_HOME is not defined. Please source upmem_env.sh.)
endif

NR_TASKLETS ?= 12

BIN_DIR := bin
__dirs := $(shell mkdir -p $(BIN_DIR))

HOST := $(BIN_DIR)/run
DPU := $(BIN_DIR)/small.dpu $(BIN_DIR)/large.dpu

all : $(HOST) $(DPU)

# Compiler flags
CFLAGS := -O3 -fopenmp
LIBS := -lm -ldl `dpu-pkg-config --cflags --libs dpu`
DPU_FLAGS := -Wall -Wextra -O3 -DNR_TASKLETS=$(NR_TASKLETS)

$(BIN_DIR)/run: host/run.cc
	gcc $(CFLAGS) host/run.cc -o $(BIN_DIR)/run $(LIBS)

$(BIN_DIR)/small.dpu: dpu/small.c
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $(BIN_DIR)/small.dpu dpu/small.c

$(BIN_DIR)/large.dpu: dpu/large.c
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $(BIN_DIR)/large.dpu dpu/large.c

clean:
	rm -rf $(BIN_DIR)
