# https://github.com/CMU-SAFARI/prim-benchmarks/tree/main 
# leveraged the above repository to create Makefile for DPU and host code compilation

DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= bin
NR_DPUS ?= 16
NR_TASKLETS ?= 16


HOST_TARGET := ${BUILDDIR}/libvectordpu.so
DPU_TARGET := ${BUILDDIR}/runtime.dpu

COMMON_INCLUDES := common
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cc)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES}
HOST_FLAGS := ${COMMON_FLAGS} -O3 `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS}
DPU_FLAGS := ${COMMON_FLAGS} -O2 -DNR_TASKLETS=${NR_TASKLETS}

all: ${HOST_TARGET} ${DPU_TARGET}

${HOST_TARGET}: ${HOST_SOURCES} ${COMMON_INCLUDES}
	$(CC) -shared -fPIC -o $@.so ${HOST_SOURCES} ${HOST_FLAGS}


${DPU_TARGET}: ${DPU_SOURCES} ${COMMON_INCLUDES}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)

test: all
	./${HOST_TARGET}