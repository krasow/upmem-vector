# Operation dispatch for DpuVector.
#
# Every op is named by its opcode from src/opcodes.jl, which the generator emits
# alongside common/opcodes.h.  The C++ wrapper switches on the same value, so
# there is one numbering rather than a table of indices to keep in step.

using .Opcodes

# ---- generic dispatch functions ----

function binary_op(a::DpuVector, b::DpuVector, op::UInt8)
    handle = retry_on_oom(() -> PolymerPIM.launch_binary(a.handle, b.handle, op))
    return DpuVector(handle)
end

function scalar_op(a::DpuVector, s::Integer, op::UInt8)
    handle = retry_on_oom(() -> PolymerPIM.launch_binary_scalar(a.handle, Int32(s), op))
    return DpuVector(handle)
end

function unary_op(a::DpuVector, op::UInt8)
    handle = retry_on_oom(() -> PolymerPIM.launch_unary(a.handle, op))
    return DpuVector(handle)
end

function reduce_op(a::DpuVector, op::UInt8)
    return retry_on_oom(() -> PolymerPIM.launch_reduction(a.handle, op))
end

"""
    reduce_lazy(v, op) -> DpuFuture

Queue a reduction without reading it.  Independent reductions left unread are
merged into a single DPU kernel pass, so prefer this (or [`sums`](@ref)) when
reducing several vectors.
"""
function reduce_lazy(a::DpuVector, op::UInt8)
    handle = retry_on_oom(() -> PolymerPIM.launch_reduction_lazy(a.handle, op))
    return DpuFuture(handle)
end

function select_op(cond::DpuVector, a::DpuVector, b::DpuVector)
    handle = retry_on_oom(() -> PolymerPIM.launch_select(cond.handle, a.handle, b.handle))
    return DpuVector(handle)
end

# ---- Base overloads: binary vector ⊕ vector ----

Base.:+(a::DpuVector, b::DpuVector) = binary_op(a, b, Opcodes.OP_ADD)
Base.:-(a::DpuVector, b::DpuVector) = binary_op(a, b, Opcodes.OP_SUB)
Base.:*(a::DpuVector, b::DpuVector) = binary_op(a, b, Opcodes.OP_MUL)
Base.div(a::DpuVector, b::DpuVector) = binary_op(a, b, Opcodes.OP_DIV)
Base.:<(a::DpuVector, b::DpuVector)  = binary_op(a, b, Opcodes.OP_LT)

# ---- Base overloads: vector ⊕ scalar / scalar ⊕ vector ----

Base.:+(a::DpuVector, s::Integer) = scalar_op(a, s, Opcodes.OP_ADD_SCALAR)
Base.:+(s::Integer, a::DpuVector) = scalar_op(a, s, Opcodes.OP_ADD_SCALAR)
Base.:-(a::DpuVector, s::Integer) = scalar_op(a, s, Opcodes.OP_SUB_SCALAR)
Base.:*(a::DpuVector, s::Integer) = scalar_op(a, s, Opcodes.OP_MUL_SCALAR)
Base.:*(s::Integer, a::DpuVector) = scalar_op(a, s, Opcodes.OP_MUL_SCALAR)
Base.div(a::DpuVector, s::Integer) = scalar_op(a, s, Opcodes.OP_DIV_SCALAR)
Base.:>>(a::DpuVector, s::Integer) = scalar_op(a, s, Opcodes.OP_ASR_SCALAR)
Base.:(==)(a::DpuVector, s::Integer) = scalar_op(a, s, Opcodes.OP_EQ_SCALAR)

# ---- Base overloads: unary ----

Base.:-(a::DpuVector)  = unary_op(a, Opcodes.OP_NEGATE)
Base.abs(a::DpuVector) = unary_op(a, Opcodes.OP_ABS)

# ---- Base overloads: reductions ----

Base.sum(v::DpuVector)     = reduce_op(v, Opcodes.OP_SUM)
Base.prod(v::DpuVector)    = reduce_op(v, Opcodes.OP_PRODUCT)
Base.minimum(v::DpuVector) = reduce_op(v, Opcodes.OP_MIN)
Base.maximum(v::DpuVector) = reduce_op(v, Opcodes.OP_MAX)

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
function apply!(a::DpuVector, b::DpuVector, op::UInt8)
    retry_on_oom(() -> PolymerPIM.var"apply_binary!"(a.handle, b.handle, op))
    return a
end

function apply!(a::DpuVector, s::Integer, op::UInt8)
    retry_on_oom(() -> PolymerPIM.var"apply_scalar!"(a.handle, Int32(s), op))
    return a
end

add!(a::DpuVector, b::DpuVector) = apply!(a, b, Opcodes.OP_ADD)
sub!(a::DpuVector, b::DpuVector) = apply!(a, b, Opcodes.OP_SUB)
mul!(a::DpuVector, b::DpuVector) = apply!(a, b, Opcodes.OP_MUL)
div!(a::DpuVector, b::DpuVector) = apply!(a, b, Opcodes.OP_DIV)

add!(a::DpuVector, s::Integer) = apply!(a, s, Opcodes.OP_ADD_SCALAR)
sub!(a::DpuVector, s::Integer) = apply!(a, s, Opcodes.OP_SUB_SCALAR)
mul!(a::DpuVector, s::Integer) = apply!(a, s, Opcodes.OP_MUL_SCALAR)
div!(a::DpuVector, s::Integer) = apply!(a, s, Opcodes.OP_DIV_SCALAR)
shr!(a::DpuVector, s::Integer) = apply!(a, s, Opcodes.OP_ASR_SCALAR)

export apply!, add!, sub!, mul!, div!, shr!

# ---- broadcasting ----
#
# The Broadcasted tree is kept lazy and lowered to a single RPN program at
# materialise time, so `a .+ b .* c` is one kernel pass by construction rather
# than three ops the runtime then has to fuse back together.  Nothing here
# depends on the fusion pass or its lookahead window.

struct DpuStyle <: Base.Broadcast.BroadcastStyle end

Base.broadcastable(v::DpuVector) = v
Base.BroadcastStyle(::Type{DpuVector}) = DpuStyle()
Base.BroadcastStyle(::DpuStyle, ::Base.Broadcast.BroadcastStyle) = DpuStyle()

# Operators reachable inside a broadcast, mapped to the expression builder.
const _BCAST_BINARY = Dict{Any,Function}(
    (+) => (+), (-) => (-), (*) => (*), div => div, (>>) => (>>),
    (==) => (==), (<) => (<), (>) => (>), (<=) => (<=), (>=) => (>=),
)
const _BCAST_UNARY = Dict{Any,Function}(
    (-) => (-), abs => abs, identity => identity,
)

# Lowering state: which vector became input(), and the operand slots assigned so
# far.  Slots are matched by object identity, so a vector used twice is loaded
# once.
mutable struct _Lowering
    primary::Union{Nothing,DpuVector}
    operands::Vector{DpuVector}
end
_Lowering() = _Lowering(nothing, DpuVector[])

function _leaf(v::DpuVector, st::_Lowering)
    if st.primary === nothing
        st.primary = v
        return input()
    end
    st.primary === v && return input()
    for (i, o) in enumerate(st.operands)
        o === v && return operand(i)
    end
    length(st.operands) + 1 <= MAX_VFUSE_INPUTS || throw(ArgumentError(
        "broadcast needs more than $MAX_VFUSE_INPUTS operand slots; split it"))
    push!(st.operands, v)
    return operand(length(st.operands))
end

_lower(v::DpuVector, st::_Lowering) = _leaf(v, st)
_lower(x::Integer, st::_Lowering) = constant(x)
_lower(e::DpuExpr, ::_Lowering) = e
_lower(x::Base.RefValue, st::_Lowering) = _lower(x[], st)

function _lower(bc::Base.Broadcast.Broadcasted, st::_Lowering)
    f = bc.f
    args = bc.args
    if length(args) == 1
        haskey(_BCAST_UNARY, f) || throw(ArgumentError(
            "$f is not supported inside a DpuVector broadcast"))
        return _BCAST_UNARY[f](_lower(args[1], st))
    elseif length(args) == 2
        # ifelse is the broadcast spelling of a per-lane select
        f === ifelse && throw(ArgumentError("ifelse needs three arguments"))
        haskey(_BCAST_BINARY, f) || throw(ArgumentError(
            "$f is not supported inside a DpuVector broadcast"))
        op = _BCAST_BINARY[f]
        a, b = args
        # An integer operand becomes an immediate rather than a pushed value.
        if b isa Integer && !(a isa Integer)
            return op(_lower(a, st), b)
        elseif a isa Integer && !(b isa Integer)
            # only the commutative/reversible ones have a scalar-first form
            f === (+) && return _lower(b, st) + a
            f === (*) && return _lower(b, st) * a
            return op(constant(a), _lower(b, st))
        end
        return op(_lower(a, st), _lower(b, st))
    elseif length(args) == 3 && f === ifelse
        return select(_lower(args[1], st), _lower(args[2], st),
                      _lower(args[3], st))
    end
    throw(ArgumentError("$f with $(length(args)) arguments is not supported " *
                        "inside a DpuVector broadcast"))
end

# Lower a whole tree to (program, primary, operands).  Deliberately not via
# Broadcast.flatten: that rewrites the tree into a synthesised closure over
# Pick{} leaves, which erases the operator identities this dispatches on.
function _lower_tree(bc::Base.Broadcast.Broadcasted)
    st = _Lowering()
    e = _lower(bc, st)
    st.primary === nothing &&
        throw(ArgumentError("broadcast contains no DpuVector"))
    return e, st.primary, st.operands
end

"""
    materialize(bc)

`a .+ b .* c` and friends: the whole expression becomes one RPN program and one
kernel pass, with no host-side intermediates.
"""
function Base.copy(bc::Base.Broadcast.Broadcasted{DpuStyle})
    e, primary, operands = _lower_tree(bc)
    return dpu_pipeline(primary, e; operands = operands)
end

"""
    dest .= expr

Writes through `dest`'s existing buffer, so other handles to it observe the
result. One kernel pass.
"""
function Base.copyto!(dest::DpuVector, bc::Base.Broadcast.Broadcasted{DpuStyle})
    e, primary, operands = _lower_tree(bc)
    length(dest) == length(primary) || throw(DimensionMismatch(
        "destination has $(length(dest)) elements, expression $(length(primary))"))
    _check_program(e, operands)
    retry_on_oom(() -> PolymerPIM.launch_pipeline_into(
        dest.handle, primary.handle, e.ops, _veclist(operands), Int32[]))
    return dest
end

# A scalar fill still goes through the same path.
Base.copyto!(dest::DpuVector, bc::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}}) =
    copyto!(dest, Base.Broadcast.broadcasted(identity, bc.f(bc.args...)))

Base.similar(v::DpuVector) = DpuVector(length(v))
Base.similar(v::DpuVector, ::Type{Int32}) = DpuVector(length(v))
Base.axes(v::DpuVector) = (Base.OneTo(length(v)),)

# ---- lazy reductions ----

"""
    sums(vectors) -> Vector{Int64}

Sum several vectors in one pass.  Queues every reduction before reading any of
them, which is what allows them to be fused into a single kernel.
"""
function sums(vs::AbstractVector{DpuVector})
    futures = [reduce_lazy(v, Opcodes.OP_SUM) for v in vs]
    return [get(f) for f in futures]
end

lazy_sum(v::DpuVector)     = reduce_lazy(v, Opcodes.OP_SUM)
lazy_prod(v::DpuVector)    = reduce_lazy(v, Opcodes.OP_PRODUCT)
lazy_minimum(v::DpuVector) = reduce_lazy(v, Opcodes.OP_MIN)
lazy_maximum(v::DpuVector) = reduce_lazy(v, Opcodes.OP_MAX)

export select_op, sums, lazy_sum, lazy_prod, lazy_minimum, lazy_maximum

# ---- RPN pipelines ----
#
# `transform` and `reduce_expr` are the Julia equivalents of the C++
# transform()/reduce() expression lambdas.  The program is built here (see
# expr.jl) and submitted through pipeline()/pipeline_reduce(), so it fuses the
# same way and also works when the library was built with JIT=0.

function _veclist(vs)
    l = PolymerPIM.DpuVecList()
    for v in vs
        PolymerPIM.var"veclist_push!"(l, v.handle)
    end
    return l
end

function _check_program(_::DpuExpr, operands)
    length(operands) <= MAX_VFUSE_INPUTS || throw(ArgumentError(
        "$(length(operands)) operands exceeds MAX_VFUSE_INPUTS ($MAX_VFUSE_INPUTS)"))
    return nothing
end

"""
    dpu_pipeline(v, e; operands=DpuVector[], scalars=Int32[]) -> DpuVector

Run the RPN program `e` over `v`, returning the elementwise result.
`input()` refers to `v`, `operand(i)` to `operands[i]`, `scalar_var(i)` to
`scalars[i]`.
"""
function dpu_pipeline(v::DpuVector, e::DpuExpr;
                  operands::AbstractVector{DpuVector} = DpuVector[],
                  scalars::AbstractVector{<:Integer} = Int32[])
    _check_program(e, operands)
    sc = Int32.(collect(scalars))
    handle = retry_on_oom(() -> PolymerPIM.launch_pipeline(
        v.handle, e.ops, _veclist(operands), sc))
    return DpuVector(handle)
end

"""
    dpu_pipeline_reduce(v, e; operands, scalars) -> DpuFuture

As [`dpu_pipeline`](@ref), but `e` must end in a reduction terminal (`sum`, `prod`,
`minimum`, `maximum`). Returns a future so independent reductions still fuse.
"""
function dpu_pipeline_reduce(v::DpuVector, e::DpuExpr;
                          operands::AbstractVector{DpuVector} = DpuVector[],
                          scalars::AbstractVector{<:Integer} = Int32[])
    _check_program(e, operands)
    isempty(e.ops) && throw(ArgumentError("empty program"))
    Opcodes.is_reduction(e.ops[end]) || throw(ArgumentError(
        "program must end in a reduction terminal (sum/prod/minimum/maximum)"))
    sc = Int32.(collect(scalars))
    handle = retry_on_oom(() -> PolymerPIM.launch_pipeline_reduce(
        v.handle, e.ops, _veclist(operands), sc))
    return DpuFuture(handle)
end

"""
    transform(f, v, operands...; scalars=Int32[]) -> DpuVector

Build an elementwise expression and run it in one fused kernel. `f` receives a
`Vector{DpuExpr}` whose first entry is `v` and whose rest are `operands`.

    transform(a, b) do x
        abs(x[1] - x[2])
    end
"""
function transform(f, v::DpuVector, operands::DpuVector...;
                   scalars::AbstractVector{<:Integer} = Int32[])
    exprs = DpuExpr[input()]
    for i in 1:length(operands)
        push!(exprs, operand(i))
    end
    return dpu_pipeline(v, f(exprs); operands = DpuVector[operands...], scalars = scalars)
end

"""
    reduce_expr(f, v, operands...; scalars=Int32[]) -> DpuFuture

As [`transform`](@ref), but `f` must return a reduction. Left unread, several of
these fuse into a single kernel pass.

    reduce_expr(a, b) do x
        sum(x[1] * x[2])          # dot product
    end
"""
function reduce_expr(f, v::DpuVector, operands::DpuVector...;
                     scalars::AbstractVector{<:Integer} = Int32[])
    exprs = DpuExpr[input()]
    for i in 1:length(operands)
        push!(exprs, operand(i))
    end
    return dpu_pipeline_reduce(v, f(exprs); operands = DpuVector[operands...],
                           scalars = scalars)
end

export dpu_pipeline, dpu_pipeline_reduce, transform, reduce_expr

# ---- K-ary argmin / argmax over whole vectors ----

"""
    argmin_of(vectors) / argmax_of(vectors) -> DpuVector

Per element, the 0-based index of the winning vector. One fused kernel pass.
"""
function argmin_of(vs::AbstractVector{DpuVector})
    isempty(vs) && throw(ArgumentError("need at least one vector"))
    handle = retry_on_oom(() -> PolymerPIM.launch_argmin_k(_veclist(vs)))
    return DpuVector(handle)
end

function argmax_of(vs::AbstractVector{DpuVector})
    isempty(vs) && throw(ArgumentError("need at least one vector"))
    handle = retry_on_oom(() -> PolymerPIM.launch_argmax_k(_veclist(vs)))
    return DpuVector(handle)
end

"""
    min_squared_distance(cols, query) -> DpuFuture

Minimum over rows of the squared euclidean distance to `query`, where `cols[j]`
holds coordinate `j` of every row. One fused pass over all columns.

`vectordpu.h` declared a C++ `min_squared_distance` but never defined it, so
this is built from the expression API instead.
"""
function min_squared_distance(cols::AbstractVector{DpuVector},
                              query::AbstractVector{<:Integer})
    isempty(cols) && throw(ArgumentError("need at least one column"))
    length(cols) == length(query) || throw(ArgumentError(
        "$(length(cols)) columns but $(length(query)) query coordinates"))
    rest = DpuVector[cols[j] for j in 2:length(cols)]
    return reduce_expr(cols[1], rest...) do x
        acc = sqr(x[1] - query[1])
        for j in 2:length(x)
            acc = acc + sqr(x[j] - query[j])
        end
        minimum(acc)
    end
end

export argmin_of, argmax_of, min_squared_distance

# ---- elementwise comparisons, via RPN ----
#
# The opcodes exist and both backends implement them, but no C++ dpu_vector
# operator wraps them, so these go through a two-operand RPN program.

for (f, builder) in ((:>, :>), (:>=, :>=), (:<=, :<=))
    @eval Base.$f(a::DpuVector, b::DpuVector) =
        transform(a, b) do x
            $builder(x[1], x[2])
        end
end

Base.:(==)(a::DpuVector, b::DpuVector) = transform(a, b) do x
    x[1] == x[2]
end

Base.:>(a::DpuVector, s::Integer) = transform(a) do x
    x[1] > s
end
Base.:>=(a::DpuVector, s::Integer) = transform(a) do x
    x[1] >= s
end
Base.:<=(a::DpuVector, s::Integer) = transform(a) do x
    x[1] <= s
end
Base.:<(a::DpuVector, s::Integer) = transform(a) do x
    x[1] < s
end

# ---- local scatter accumulators ----

"""
    DpuLocalVector(n; reduce_op = :sum)

A small per-DPU accumulator array in WRAM, the target of a scatter program.
`Array(l)` gathers every DPU's copy and merges them with `reduce_op`.
"""
mutable struct DpuLocalVector
    handle::Any
    len::Int
    reduce_op::Symbol
end

function DpuLocalVector(n::Integer; reduce_op::Symbol = :sum)
    reduce_op in LOCAL_REDUCE_OPS || throw(ArgumentError(
        "reduce_op must be one of $(LOCAL_REDUCE_OPS)"))
    idx = findfirst(==(reduce_op), LOCAL_REDUCE_OPS) - 1
    handle = retry_on_oom(() -> PolymerPIM.local_alloc(Int32(n), Int32(idx)))
    return DpuLocalVector(handle, Int(n), reduce_op)
end

Base.length(l::DpuLocalVector) = l.len

function Base.Array(l::DpuLocalVector)
    out = Vector{Int32}(undef, l.len)
    retry_on_oom(() -> PolymerPIM.var"local_to_cpu!"(l.handle, out))
    return out
end

"""
    scatter!(locals, v, program; operands, scalars)

Run a scatter `program` (see [`scatter_program`](@ref)) over `v`, accumulating
into `locals`. Slot numbers in the program's `LocalReduce` entries are 0-based
indices into `locals`.
"""
function scatter!(locals::AbstractVector{DpuLocalVector}, v::DpuVector,
                  program::DpuExpr;
                  operands::AbstractVector{DpuVector} = DpuVector[],
                  scalars::AbstractVector{<:Integer} = Int32[])
    isempty(program.ops) && return locals
    length(locals) <= MAX_LOCAL_SCRATCH_VECTORS || throw(ArgumentError(
        "$(length(locals)) local vectors exceeds MAX_LOCAL_SCRATCH_VECTORS " *
        "($MAX_LOCAL_SCRATCH_VECTORS); WRAM has room for no more, and the " *
        "extras would silently read back as zeros"))
    _check_program(program, operands)
    ll = PolymerPIM.DpuLocalList()
    for l in locals
        PolymerPIM.var"locallist_push!"(ll, l.handle)
    end
    sc = Int32.(collect(scalars))
    retry_on_oom(() -> PolymerPIM.launch_pipeline_scatter(
        v.handle, program.ops, _veclist(operands), sc, ll))
    return locals
end

export DpuLocalVector, scatter!
