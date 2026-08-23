# PolymerPIM.jl -- convenience Makefile
#
# Prerequisites:
#   1. Source the UPMEM environment:
#        source /usr/upmem_env.sh
#
#   2. Install Julia (>= 1.10) and the CxxWrap package:
#        julia -e 'using Pkg; Pkg.add("CxxWrap")'
#
# Then from this directory:
#   make          # build and install the C++ wrapper into lib/wrapper/PolymerPIM/
#   make test     # run Julia tests
#   make config   # show the recorded configuration of the install
#   make clean    # remove build artefacts

JULIA       ?= julia
LOCAL_ROOT := $(abspath ../install)
ifeq ($(origin POLYMERPIM_ROOT), undefined)
POLYMERPIM_ROOT := $(LOCAL_ROOT)
PREPARE_CPP ?= 1
else
PREPARE_CPP ?= 0
endif

# Where deps/build.jl installs the wrapper, next to its config stamps.
INSTALL_DIR := lib/wrapper/PolymerPIM

.PHONY: all build wrapper install-cpp test clean config

all: build

ifeq ($(PREPARE_CPP),1)
build: install-cpp wrapper
wrapper: install-cpp
else
build: wrapper
endif

install-cpp:
	$(MAKE) -C .. install-cpp DESTDIR=$(POLYMERPIM_ROOT) \
		PIPELINE=1 JIT=1 JULIA=$(JULIA)

# Precompilation is off for the instantiate step: it loads PolymerPIM, which
# binds the wrapper deps/build.jl has not rebuilt yet, so a newly added wrapper
# method would fail here before the library that has it exists.
wrapper:
	JULIA_PKG_PRECOMPILE_AUTO=0 $(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	POLYMERPIM_ROOT=$(POLYMERPIM_ROOT) $(JULIA) --project=. deps/build.jl

# JIT shells out to the DPU compiler per kernel, which greets each invocation
# with a debian10 deprecation notice; silence it so test output stays readable.
test: build
	UPMEM_NO_OS_WARNING=1 $(JULIA) --project=. test/runtests.jl

# Print the recorded configuration of the installed wrapper.
config:
	@cat $(INSTALL_DIR)/build.config $(INSTALL_DIR)/install.config

clean:
	rm -rf lib/wrapper/build $(INSTALL_DIR)
