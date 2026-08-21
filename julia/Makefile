# PolymerPIM.jl -- convenience Makefile
#
# Prerequisites:
#   1. Source the UPMEM environment:
#        source /usr/upmem_env.sh
#
#   2. Build and install upmem-vector (from the upmem-vector repo):
#        make install BACKEND=hw JIT=1
#      This installs to ../vectordpu/ by default.
#
#   3. Install Julia (>= 1.10) and the CxxWrap package:
#        julia -e 'using Pkg; Pkg.add("CxxWrap")'
#
# Then from this directory:
#   make          # build and install the C++ wrapper into lib/wrapper/PolymerPIM/
#   make test     # run Julia tests
#   make config   # show the recorded configuration of the install
#   make clean    # remove build artefacts

JULIA       ?= julia
# Install prefix of the C++ library; matches the source Makefile's DESTDIR.
VECTORDPU_DIR ?= $(abspath ../../vectordpu)

# Where deps/build.jl installs the wrapper, next to its config stamps.
INSTALL_DIR := lib/wrapper/PolymerPIM

.PHONY: all build test clean config

all: build

# Precompilation is off for the instantiate step: it loads PolymerPIM, which
# binds the wrapper deps/build.jl has not rebuilt yet, so a newly added wrapper
# method would fail here before the library that has it exists.
build:
	JULIA_PKG_PRECOMPILE_AUTO=0 $(JULIA) --project=. -e 'using Pkg; Pkg.instantiate()'
	VECTORDPU_DIR=$(VECTORDPU_DIR) $(JULIA) --project=. deps/build.jl

# JIT shells out to the DPU compiler per kernel, which greets each invocation
# with a debian10 deprecation notice; silence it so test output stays readable.
test: build
	UPMEM_NO_OS_WARNING=1 VECTORDPU_DIR=$(VECTORDPU_DIR) $(JULIA) --project=. test/runtests.jl

# Print the recorded configuration of the installed wrapper.
config:
	@cat $(INSTALL_DIR)/build.config $(INSTALL_DIR)/install.config

clean:
	rm -rf lib/wrapper/build $(INSTALL_DIR)
