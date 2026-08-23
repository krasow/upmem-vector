using CxxWrap

const WRAPPER_DIR = normpath(joinpath(@__DIR__, "..", "lib", "wrapper"))
const BUILD_DIR = joinpath(WRAPPER_DIR, "build")
# Where the finished library lands, next to the stamps describing it.
const INSTALL_DIR = joinpath(WRAPPER_DIR, "PolymerPIM")
const WRAPPER_LIB = "libpolymerpim_wrapper.so"

jlcxx_prefix = CxxWrap.prefix_path()
julia_prefix = joinpath(Sys.BINDIR, "..")

# The repository-local C++ install prefix, overridable for packaged installs.
const DEFAULT_PREFIX = normpath(joinpath(@__DIR__, "..", "..", "install"))
polymerpim_root = abspath(get(ENV, "POLYMERPIM_ROOT", DEFAULT_PREFIX))

isdir(polymerpim_root) || error("""
    PolymerPIM not installed at $polymerpim_root. From the source tree:
        make install PIPELINE=1 JIT=1 BACKEND=hw
    or set POLYMERPIM_ROOT to an existing install.""")
prefix_share = joinpath(polymerpim_root, "share", "polymerpim")

# The prefix is a copy and can lag the source tree it came from.  Once the
# wrapper is linked the two are indistinguishable, so check now.
source_config = joinpath(@__DIR__, "..", "..", "build.config")
if !isfile(joinpath(prefix_share, "build.config"))
    @warn "$polymerpim_root records no build.config; reinstall the C++ library to record one"
elseif isfile(source_config) &&
       read(source_config, String) != read(joinpath(prefix_share, "build.config"), String)
    @warn "$polymerpim_root is stale relative to its source tree; run `make install` there first"
end

@info "Building PolymerPIM C++ wrapper" jlcxx_prefix polymerpim_root

mkpath(BUILD_DIR)
cd(BUILD_DIR) do
    run(`cmake $(WRAPPER_DIR)
        -DCMAKE_PREFIX_PATH=$(jlcxx_prefix)
        -DJulia_PREFIX=$(julia_prefix)
        -DPOLYMERPIM_ROOT=$(polymerpim_root)
        -DCMAKE_BUILD_TYPE=Release`)
    run(`cmake --build . --config Release`)
end

mkpath(INSTALL_DIR)
cp(joinpath(BUILD_DIR, WRAPPER_LIB), joinpath(INSTALL_DIR, WRAPPER_LIB); force = true)

# build.config is the flag set the loaded library must still report at load time;
# install.config is provenance, the prefix's own plus how the wrapper was configured.
cp(joinpath(prefix_share, "build.config"), joinpath(INSTALL_DIR, "build.config"); force = true)
open(joinpath(INSTALL_DIR, "install.config"), "w") do io
    prefix_install = joinpath(prefix_share, "install.config")
    isfile(prefix_install) && write(io, read(prefix_install))
    println(io, "POLYMERPIM_ROOT=", polymerpim_root)
    println(io, "WRAPPER_BUILT=", Libc.strftime("%Y-%m-%dT%H:%M:%S%z", time()))
    println(io, "JULIA_VERSION=", VERSION)
    println(io, "CXXWRAP_PREFIX=", jlcxx_prefix)
end

@info "PolymerPIM C++ wrapper installed" INSTALL_DIR
