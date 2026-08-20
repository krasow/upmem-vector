module UpmemVector

using CxxWrap

# Path to the compiled wrapper shared library (without extension)
const _wrapper_lib = joinpath(@__DIR__, "..", "lib", "wrapper", "build", "libupmemvector_wrapper")

@wrapmodule(() -> _wrapper_lib)

function __init__()
    @initcxx
    # PIPELINE=0 / JIT=0 exist so the C++ library can benchmark the
    # alternatives; this package targets the configuration you would actually
    # deploy, and the op set it binds only exists there.  Checked on every load
    # so a relinked library cannot be masked by a precompile cache.
    if !UpmemVector.built_with_pipeline() || !UpmemVector.built_with_jit()
        error("""
              UpmemVector.jl requires libvectordpu built with PIPELINE=1 JIT=1 \
              (found PIPELINE=$(Int(UpmemVector.built_with_pipeline())) \
              JIT=$(Int(UpmemVector.built_with_jit()))).

              Rebuild and reinstall the C++ library, then rebuild the wrapper:
                make PIPELINE=1 JIT=1 BACKEND=hw install
                make -C julia clean build
              """)
    end
    atexit() do
        try
            GC.gc(true) # we need to ensure all vectors are destructed before cleanup is called
            UpmemVector.cleanup()
        catch e # ignore errors during shutdown
        end
    end
end

"""
    sync()

Synchronize all DPUs: blocks until all pending operations on all vectors
complete.
"""
function sync()
    UpmemVector.dpu_sync()
end

"""
    retry_on_oom(f)

Executes function `f`. If a DPU OOM exception is caught, triggers Julia GC
and retries once.
"""
function retry_on_oom(f)
    try
        return f()
    catch e
        # CxxWrap throws exceptions as CxxException.
        # We check the message for "DPU OOM".
        if occursin("DPU OOM", sprint(showerror, e))
            @warn "DPU OOM detected. Syncing and triggering GC..."
            sync()     # Flush event queue and wait for DPUs
            GC.gc(true) # Major GC
            yield()
            return f()
        else
            rethrow(e)
        end
    end
end

# Slot limits read from the library itself, so they cannot drift from the
# configuration it was compiled with.
const MAX_VFUSE_INPUTS = Int(UpmemVector.limit_operands())
const MAX_PIPELINE_SCALARS = Int(UpmemVector.limit_scalars())
const MAX_LOCAL_SCRATCH_VECTORS = Int(UpmemVector.limit_locals())

include("opcodes.jl")
include("types.jl")
include("expr.jl")
include("operations.jl")
include("display.jl")

export DpuVector, DpuFuture, fence, sync
export MAX_VFUSE_INPUTS, MAX_PIPELINE_SCALARS, MAX_LOCAL_SCRATCH_VECTORS

end # module UpmemVector
