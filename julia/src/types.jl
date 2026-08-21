# DPUVector -- Julia wrapper around the CxxWrap-managed dpu_vector<int32_t>

"""
    DPUVector(data::AbstractVector{<:Integer})
    DPUVector(n::Integer)

A 1-D vector stored in UPMEM DPU memory.

Construct from a Julia array to transfer data to the DPUs, or pass an integer
to allocate an uninitialised vector of that length.

# Examples
```julia
v = DPUVector(Int32[1, 2, 3, 4])
w = DPUVector(1024)                 # uninitialised, length 1024
```
"""
mutable struct DPUVector
    handle::PolymerPIM.DpuVectorInt32   # CxxWrap-managed C++ object
    len::Int64

    function DPUVector(handle::PolymerPIM.DpuVectorInt32)
        v = new(handle, Int64(PolymerPIM.cpp_length(handle)))
        return v
    end
end

# Construct from a Julia vector -- transfer to DPU memory.  Handed over as-is:
# it is already the contiguous buffer the wrapper wants, and `collect` would
# copy the lot (~200ms per 512MB) first.  The wrapper fences before returning.
function DPUVector(data::Vector{Int32})
    handle = retry_on_oom(() -> PolymerPIM.from_cpu_int32(data))
    return DPUVector(handle)
end

# Views and other lazy Int32 vectors have to be materialised first.
function DPUVector(data::AbstractVector{Int32})
    return DPUVector(collect(Int32, data))
end

# Accept any integer array by converting to Int32
function DPUVector(data::AbstractVector{<:Integer})
    return DPUVector(convert(Vector{Int32}, data))
end

# Allocate an uninitialised DPU vector of length n.  CxxWrap exposes the bound
# `constructor<uint32_t>()` as the type itself, not as a cpp_alloc_* helper.
function DPUVector(n::Integer)
    n >= 0 || throw(ArgumentError("length must be non-negative, got $n"))
    handle = retry_on_oom(() -> PolymerPIM.DpuVectorInt32(UInt32(n)))
    return DPUVector(handle)
end

# ---- Conversions: DPU -> Julia ----

"""
    Array(v::DPUVector) -> Vector{Int32}

Transfer DPU vector contents back to the host as a Julia `Vector{Int32}`.
"""
function Base.Array(v::DPUVector)
    out = Vector{Int32}(undef, v.len)
    retry_on_oom(() -> PolymerPIM.to_cpu!(v.handle, out))
    return out
end

Base.Vector(v::DPUVector) = Array(v)
Base.collect(v::DPUVector) = Array(v)

# ---- Basic queries ----

Base.length(v::DPUVector) = v.len
Base.size(v::DPUVector) = (v.len,)
Base.eltype(::DPUVector) = Int32

# Scalar indexing (requires full transfer -- use sparingly)
function Base.getindex(v::DPUVector, i::Int)
    @boundscheck 1 <= i <= v.len || throw(BoundsError(v, i))
    return Array(v)[i]
end

"""
    fence(v::DPUVector)

Explicitly synchronize: block until all pending DPU operations on `v` complete.
"""
function fence(v::DPUVector)
    retry_on_oom(() -> PolymerPIM.dpu_fence(v.handle))
end

export fence

"""
    release!(v::DPUVector)

Free `v`'s DPU memory now instead of waiting for the GC to collect it.  The GC
cannot see MRAM pressure, so a loop that allocates a fresh vector each pass can
exhaust the DPUs while the dead ones are still unreclaimed; C++ avoids this by
destroying each `dpu_vector` at scope exit.  Using `v` afterwards is invalid.
"""
function release!(v::DPUVector)
    finalize(v.handle)
    return nothing
end

export release!

# ---- DpuFuture -- a queued reduction whose value has not been read ----

"""
    DpuFuture

A queued reduction, unread.  Leave several unread and they merge into one
kernel pass; `f[]`, `get(f)` or `fetch(f)` forces them.
"""
mutable struct DpuFuture
    handle::PolymerPIM.DpuFutureInt32
end

"""
    get(f::DpuFuture) -> Int64

Read a queued reduction, blocking until the DPUs have produced it.
"""
Base.get(f::DpuFuture) = retry_on_oom(() -> PolymerPIM.cpp_get(f.handle))

# `f[]` as for any value holder; `fetch` is the Julia future spelling.
Base.getindex(f::DpuFuture) = get(f)
Base.fetch(f::DpuFuture) = get(f)

export DpuFuture
