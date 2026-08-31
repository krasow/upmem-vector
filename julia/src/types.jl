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

"""
    PolymerPIM.zeros(Int32, n) -> DpuZeros

`n` zeros that exist only as the additive identity: `zeros(Int32, n) .+ x`
lowers to the program `x` alone would give, so an accumulator loop can start at
its first element. Nothing is allocated until a `DPUVector` reads them.
"""
struct DpuZeros
    length::Int64

    function DpuZeros(n::Integer)
        n >= 0 || throw(ArgumentError("length must be non-negative, got $n"))
        return new(Int64(n))
    end
end

zeros(::Type{Int32}, n::Integer) = DpuZeros(n)

Base.length(z::DpuZeros) = z.length
Base.size(z::DpuZeros) = (z.length,)
Base.eltype(::DpuZeros) = Int32

# Dropped as the tree is built, not during lowering: folding a runtime scalar
# that happens to be 0 would split the JIT program shared across its values.
Base.broadcasted(::typeof(+), ::DpuZeros, x) = x
Base.broadcasted(::typeof(+), x, ::DpuZeros) = x

"""
    PolymerPIM.fill(Int32, n, value) -> DPUVector

`n` copies of `value`, written by the DPUs rather than staged through a host
buffer.
"""
function fill(::Type{Int32}, n::Integer, value::Integer)
    n >= 0 || throw(ArgumentError("length must be non-negative, got $n"))
    handle = retry_on_oom(() -> PolymerPIM.fill_int32(Int64(n), Int32(value)))
    return DPUVector(handle)
end

# Reading them is what materialises them.
DPUVector(z::DpuZeros) = fill(Int32, z.length, Int32(0))

# Only `+` can drop the zeros for free, so every other use materialises rather
# than failing on a type that has no storage.
Base.broadcastable(z::DpuZeros) = DPUVector(z)
for f in (:sum, :prod, :minimum, :maximum, :findmax, :findmin, :argmin, :argmax)
    @eval Base.$f(z::DpuZeros) = Base.$f(DPUVector(z))
end

export DpuZeros

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
    value::Int64
    resolved::Bool

    DpuFuture(handle::PolymerPIM.DpuFutureInt32) = new(handle, 0, false)
end

"""
    get(f::DpuFuture) -> Int64

Read a queued reduction, blocking until the DPUs have produced it.  Kept, so a
second read transfers nothing.
"""
function Base.get(f::DpuFuture)
    f.resolved && return f.value
    f.value = retry_on_oom(() -> PolymerPIM.cpp_get(f.handle))
    f.resolved = true
    return f.value
end

# `f[]` as for any value holder; `fetch` is the Julia future spelling.
Base.getindex(f::DpuFuture) = get(f)
Base.fetch(f::DpuFuture) = get(f)

# A future stands in for its value.  Both operands were queued before the read,
# so this costs no fusion.
Base.convert(::Type{T}, f::DpuFuture) where {T<:Number} = convert(T, get(f))
Base.Int64(f::DpuFuture) = get(f)

for op in (:+, :-, :*, :/, :÷, :%, :^, :min, :max)
    @eval Base.$op(f::DpuFuture, g::DpuFuture) = Base.$op(get(f), get(g))
    @eval Base.$op(f::DpuFuture, x::Number) = Base.$op(get(f), x)
    @eval Base.$op(x::Number, f::DpuFuture) = Base.$op(x, get(f))
end

# Base derives `>` and `>=` from these.
for op in (:(==), :<, :<=, :isless)
    @eval Base.$op(f::DpuFuture, g::DpuFuture) = Base.$op(get(f), get(g))
    @eval Base.$op(f::DpuFuture, x::Number) = Base.$op(get(f), x)
    @eval Base.$op(x::Number, f::DpuFuture) = Base.$op(x, get(f))
end

Base.:-(f::DpuFuture) = -get(f)
Base.abs(f::DpuFuture) = abs(get(f))
# `==` reads, so `hash` must too.
Base.hash(f::DpuFuture, h::UInt) = hash(get(f), h)

export DpuFuture
