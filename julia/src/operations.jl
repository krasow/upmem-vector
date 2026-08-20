# Modular opcode-based operation dispatch for DpuVector
#
# Each operation category has its own enum that maps 1:1 to the C++ OpEntry
# lookup tables in wrapper.cpp.  Julia Base overloads simply call the
# appropriate launch_* function with the right enum index.

# ---- operation enums (indices match the C++ OpEntry arrays) ----

module Ops

@enum BinaryOp::Int32 begin
    BINARY_ADD = 0
    BINARY_SUB = 1
    BINARY_MUL = 2
    BINARY_DIV = 3
    BINARY_LT  = 4
end

@enum ScalarOp::Int32 begin
    SCALAR_ADD = 0
    SCALAR_SUB = 1
    SCALAR_MUL = 2
    SCALAR_DIV = 3
    SCALAR_ASR = 4
    SCALAR_EQ  = 5
end

@enum UnaryOp::Int32 begin
    UNARY_NEGATE = 0
    UNARY_ABS    = 1
end

@enum ReductionOp::Int32 begin
    REDUCE_MIN     = 0
    REDUCE_MAX     = 1
    REDUCE_SUM     = 2
    REDUCE_PRODUCT = 3
end

end # module Ops

using .Ops

export Ops

# ---- generic dispatch functions ----

function binary_op(a::DpuVector, b::DpuVector, op::Ops.BinaryOp)
    handle = retry_on_oom(() -> UpmemVector.launch_binary(a.handle, b.handle, Int32(op)))
    return DpuVector(handle)
end

function scalar_op(a::DpuVector, s::Integer, op::Ops.ScalarOp)
    handle = retry_on_oom(() -> UpmemVector.launch_binary_scalar(a.handle, Int32(s), Int32(op)))
    return DpuVector(handle)
end

function unary_op(a::DpuVector, op::Ops.UnaryOp)
    handle = retry_on_oom(() -> UpmemVector.launch_unary(a.handle, Int32(op)))
    return DpuVector(handle)
end

function reduce_op(a::DpuVector, op::Ops.ReductionOp)
    return retry_on_oom(() -> UpmemVector.launch_reduction(a.handle, Int32(op)))
end

"""
    reduce_lazy(v, op) -> DpuFuture

Queue a reduction without reading it.  Independent reductions left unread are
merged into a single DPU kernel pass, so prefer this (or [`sums`](@ref)) when
reducing several vectors.
"""
function reduce_lazy(a::DpuVector, op::Ops.ReductionOp)
    handle = retry_on_oom(() -> UpmemVector.launch_reduction_lazy(a.handle, Int32(op)))
    return DpuFuture(handle)
end

function select_op(cond::DpuVector, a::DpuVector, b::DpuVector)
    handle = retry_on_oom(() -> UpmemVector.launch_select(cond.handle, a.handle, b.handle))
    return DpuVector(handle)
end

# ---- Base overloads: binary vector ⊕ vector ----

Base.:+(a::DpuVector, b::DpuVector) = binary_op(a, b, Ops.BINARY_ADD)
Base.:-(a::DpuVector, b::DpuVector) = binary_op(a, b, Ops.BINARY_SUB)
Base.:*(a::DpuVector, b::DpuVector) = binary_op(a, b, Ops.BINARY_MUL)
Base.div(a::DpuVector, b::DpuVector) = binary_op(a, b, Ops.BINARY_DIV)
Base.:<(a::DpuVector, b::DpuVector)  = binary_op(a, b, Ops.BINARY_LT)

# ---- Base overloads: vector ⊕ scalar / scalar ⊕ vector ----

Base.:+(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_ADD)
Base.:+(s::Integer, a::DpuVector) = scalar_op(a, s, Ops.SCALAR_ADD)
Base.:-(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_SUB)
Base.:*(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_MUL)
Base.:*(s::Integer, a::DpuVector) = scalar_op(a, s, Ops.SCALAR_MUL)
Base.div(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_DIV)
Base.:>>(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_ASR)
Base.:(==)(a::DpuVector, s::Integer) = scalar_op(a, s, Ops.SCALAR_EQ)

# ---- Base overloads: unary ----

Base.:-(a::DpuVector)  = unary_op(a, Ops.UNARY_NEGATE)
Base.abs(a::DpuVector) = unary_op(a, Ops.UNARY_ABS)

# ---- Base overloads: reductions ----

Base.sum(v::DpuVector)     = reduce_op(v, Ops.REDUCE_SUM)
Base.prod(v::DpuVector)    = reduce_op(v, Ops.REDUCE_PRODUCT)
Base.minimum(v::DpuVector) = reduce_op(v, Ops.REDUCE_MIN)
Base.maximum(v::DpuVector) = reduce_op(v, Ops.REDUCE_MAX)

# ---- in-place operations ----
#
# These write through the existing DPU buffer.  Chaining them is the
# memory-frugal way to build an accumulator, since no intermediate is
# allocated.

"""
    add!(a, b) / sub!(a, b) / mul!(a, b) / div!(a, b)

Apply an operation to `a` in place. `b` may be a `DpuVector` or an integer.
Returns `a`.
"""
function apply!(a::DpuVector, b::DpuVector, op::Ops.BinaryOp)
    op in (Ops.BINARY_ADD, Ops.BINARY_SUB, Ops.BINARY_MUL, Ops.BINARY_DIV) ||
        throw(ArgumentError("no in-place form for $op"))
    retry_on_oom(() -> UpmemVector.var"apply_binary!"(a.handle, b.handle, Int32(op)))
    return a
end

function apply!(a::DpuVector, s::Integer, op::Ops.ScalarOp)
    op == Ops.SCALAR_EQ && throw(ArgumentError("no in-place form for $op"))
    retry_on_oom(() -> UpmemVector.var"apply_scalar!"(a.handle, Int32(s), Int32(op)))
    return a
end

add!(a::DpuVector, b::DpuVector) = apply!(a, b, Ops.BINARY_ADD)
sub!(a::DpuVector, b::DpuVector) = apply!(a, b, Ops.BINARY_SUB)
mul!(a::DpuVector, b::DpuVector) = apply!(a, b, Ops.BINARY_MUL)
div!(a::DpuVector, b::DpuVector) = apply!(a, b, Ops.BINARY_DIV)

add!(a::DpuVector, s::Integer) = apply!(a, s, Ops.SCALAR_ADD)
sub!(a::DpuVector, s::Integer) = apply!(a, s, Ops.SCALAR_SUB)
mul!(a::DpuVector, s::Integer) = apply!(a, s, Ops.SCALAR_MUL)
div!(a::DpuVector, s::Integer) = apply!(a, s, Ops.SCALAR_DIV)
shr!(a::DpuVector, s::Integer) = apply!(a, s, Ops.SCALAR_ASR)

export apply!, add!, sub!, mul!, div!, shr!

# ---- broadcasting ----
#
# `a .+ b` is spelled the same as `a + b` on a DpuVector: every operation is
# already elementwise, so broadcasting just forwards.  This keeps idiomatic
# Julia working without materialising a lazy Broadcasted object on the host.

struct DpuStyle <: Base.Broadcast.BroadcastStyle end

Base.broadcastable(v::DpuVector) = v
Base.BroadcastStyle(::Type{DpuVector}) = DpuStyle()
Base.BroadcastStyle(::DpuStyle, ::Base.Broadcast.BroadcastStyle) = DpuStyle()

Base.broadcasted(::DpuStyle, ::typeof(+), a, b) = a + b
Base.broadcasted(::DpuStyle, ::typeof(-), a, b) = a - b
Base.broadcasted(::DpuStyle, ::typeof(*), a, b) = a * b
Base.broadcasted(::DpuStyle, ::typeof(div), a, b) = div(a, b)
Base.broadcasted(::DpuStyle, ::typeof(>>), a, s) = a >> s
Base.broadcasted(::DpuStyle, ::typeof(<), a, b) = a < b
Base.broadcasted(::DpuStyle, ::typeof(==), a, s) = a == s
Base.broadcasted(::DpuStyle, ::typeof(-), a) = -a
Base.broadcasted(::DpuStyle, ::typeof(abs), a) = abs(a)

# ---- lazy reductions ----

"""
    sums(vectors) -> Vector{Int64}

Sum several vectors in one pass.  Queues every reduction before reading any of
them, which is what allows them to be fused into a single kernel.
"""
function sums(vs::AbstractVector{DpuVector})
    futures = [reduce_lazy(v, Ops.REDUCE_SUM) for v in vs]
    return [get(f) for f in futures]
end

lazy_sum(v::DpuVector)     = reduce_lazy(v, Ops.REDUCE_SUM)
lazy_prod(v::DpuVector)    = reduce_lazy(v, Ops.REDUCE_PRODUCT)
lazy_minimum(v::DpuVector) = reduce_lazy(v, Ops.REDUCE_MIN)
lazy_maximum(v::DpuVector) = reduce_lazy(v, Ops.REDUCE_MAX)

export select_op, sums, lazy_sum, lazy_prod, lazy_minimum, lazy_maximum
