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

"""
    reduce_lazy(v, op) -> DpuFuture

Queue a reduction without reading it -- what `sum` and friends are built on.
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

# Futures, not numbers: left unread, independent reductions share a kernel.
Base.sum(v::DpuVector)     = reduce_lazy(v, Opcodes.OP_SUM)
Base.prod(v::DpuVector)    = reduce_lazy(v, Opcodes.OP_PRODUCT)
Base.minimum(v::DpuVector) = reduce_lazy(v, Opcodes.OP_MIN)
Base.maximum(v::DpuVector) = reduce_lazy(v, Opcodes.OP_MAX)

# `sum(f, v)` is the one-pass spelling: f arrives as a function, so there is no
# intermediate to materialise the way `sum(f.(v))` has.  f is traced once over a
# DpuExpr, and the terminal is appended to the program it builds.
for (f, terminal) in ((:sum, :sum), (:prod, :prod),
                      (:minimum, :minimum), (:maximum, :maximum))
    @eval Base.$f(f, v::DpuVector) =
        dpu_pipeline_reduce(v, $terminal(_trace(f)))
end

const MAPREDUCE_TERMINALS = Dict{Any,Function}(
    (+) => sum, (*) => prod, min => minimum, max => maximum)

# ---- Base overloads: which element won ----
#
# Two DPU passes: the extreme value, then the lowest index holding it.  Needs
# `global_index`, since a kernel's own index restarts on every shard.  Without
# these Base falls back to iterating, which asks for `keys`.

# Non-winners take a sentinel above every index, so the min is the first winner
# -- Base's tie too.  Value and sentinel are runtime scalars, so every call and
# every length share one compiled kernel.
_arg_index_program() =
    minimum(select(eq_var(input(), 1), global_index(), scalar_var(2)))

function _arg_reduce(v::DpuVector, want_max::Bool)
    length(v) > 0 || throw(ArgumentError("collection must be non-empty"))
    best = (want_max ? maximum(v) : minimum(v))[]
    index = get(dpu_pipeline_reduce(v, _arg_index_program();
                                    scalars = Int32[best, length(v)]))
    return best, Int(index) + 1   # the kernel counts from 0
end

"""
    findmax(v::DpuVector) -> (value, index)
    findmin(v::DpuVector) -> (value, index)

The extreme value and the first index holding it, as Base's do. The value is an
`Int64`, the type a DPU reduction returns, not the vector's `Int32`.

Two DPU passes; only scalars come back, the vector stays put. [`maximum`](@ref)
is one pass, so prefer it when the position is not needed.
"""
Base.findmax(v::DpuVector) = _arg_reduce(v, true)
Base.findmin(v::DpuVector) = _arg_reduce(v, false)

"""
    argmax(v::DpuVector) -> Int
    argmin(v::DpuVector) -> Int

The first index holding the extreme value; see [`findmax`](@ref).
"""
Base.argmax(v::DpuVector) = _arg_reduce(v, true)[2]
Base.argmin(v::DpuVector) = _arg_reduce(v, false)[2]

"""
    mapreduce(f, op, v::DpuVector)

One kernel pass: `f` is traced into the program and `op` becomes its reduction
terminal. `op` must be `+`, `*`, `min` or `max`.
"""
function Base.mapreduce(f, op, v::DpuVector)
    terminal = get(MAPREDUCE_TERMINALS, op, nothing)
    terminal === nothing && throw(ArgumentError(
        "mapreduce over a DpuVector needs op in (+, *, min, max), got $op"))
    return get(dpu_pipeline_reduce(v, terminal(_trace(f))))
end

# Trace a host function over the expression builders.  Anything outside the
# supported op set raises here rather than silently falling back to the host.
function _trace(f)
    e = f(input())
    e isa DpuExpr || throw(ArgumentError(
        "$f did not build a DpuVector expression (got $(typeof(e)))"))
    return e
end

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
    inlined::Vector{Any}    # the lazy values folded into this program
end
_Lowering() = _Lowering(nothing, DpuVector[], Any[])

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
    if f isa LaneArg   # any number of lanes, each possibly an expression
        lanes = DpuExpr[_lower(a, st) for a in args]
        return f isa LaneArg{true} ? argmax(lanes) : argmin(lanes)
    end
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
#
# @noinline is load-bearing on Julia 1.11.3: when a caller discards `primary`,
# inlining this lets SROA drop `_leaf`'s store to st.primary while keeping the
# check below, so a perfectly good broadcast reports "contains no DpuVector".
@noinline function _lower_tree(bc::Base.Broadcast.Broadcasted;
                               consume::Bool = true)
    st = _Lowering()
    e = _lower(bc, st)
    st.primary === nothing &&
        throw(ArgumentError("broadcast contains no DpuVector"))
    # Whatever was folded in needs no run of its own.  Not marked when the
    # program is only being inspected, since nothing is submitted.
    consume && for x in st.inlined
        x.consumed = true
    end
    return e, st.primary, st.operands
end

# ---- lazy results ----
#
# `a .+ b` is the program, not the vector.  Inlined where it is used, so a chain
# of statements is still one pass and a reduction over one has no intermediate.

"""
    DpuLazy

An expression built but not run. `Array`, `DpuVector`, indexing, `fence` or a
reduction runs it; using it inside another expression inlines it instead.
"""
mutable struct DpuLazy
    bc::Base.Broadcast.Broadcasted
    len::Int
    forced::Any     # the DpuVector once it has been run, so it runs once
    consumed::Bool  # inlined into a submitted program, so it needs no run of its own
end

# Registered weakly so `sync()` can run what nothing else will: a value the
# caller kept, as opposed to a step on the way to one.
const _LAZY_REGISTRY = WeakRef[]

function DpuLazy(bc::Base.Broadcast.Broadcasted, len::Int)
    x = DpuLazy(bc, len, nothing, false)
    push!(_LAZY_REGISTRY, WeakRef(x))
    return x
end

Base.copy(bc::Base.Broadcast.Broadcasted{DpuStyle}) = DpuLazy(bc, _bclength(bc))

# The element count is the primary vector's, found without lowering anything.
_bclength(x::DpuVector) = length(x)
_bclength(x::DpuLazy) = x.len
_bclength(::Any) = nothing
function _bclength(bc::Base.Broadcast.Broadcasted)
    for a in bc.args
        n = _bclength(a)
        n === nothing || return n
    end
    return nothing
end

Base.length(x::DpuLazy) = x.len
Base.size(x::DpuLazy) = (x.len,)
Base.eltype(::DpuLazy) = Int32
Base.axes(x::DpuLazy) = (Base.OneTo(x.len),)
Base.broadcastable(x::DpuLazy) = x
Base.BroadcastStyle(::Type{DpuLazy}) = DpuStyle()

# Inlined where it is used, unless already run -- then that result is cheaper.
function _lower(x::DpuLazy, st::_Lowering)
    x.forced === nothing || return _leaf(x.forced, st)
    push!(st.inlined, x)
    return _lower(x.bc, st)
end

"""
    DpuVector(x::DpuLazy)

Run `x`, keeping the result on the DPUs. [`fence`](@ref) says the same thing
without reading as a conversion.
"""
function DpuVector(x::DpuLazy)
    x.forced === nothing || return x.forced
    e, primary, operands = _lower_tree(x.bc)
    x.forced = dpu_pipeline(primary, e; operands = operands)
    return x.forced
end

Base.Array(x::DpuLazy) = Array(DpuVector(x))
Base.Vector(x::DpuLazy) = Array(x)
Base.collect(x::DpuLazy) = Array(x)
Base.getindex(x::DpuLazy, i::Integer) = Array(x)[i]

"""
    fence(x::DpuLazy)

Run `x`, block until it is done, return it. `fence(v::DpuVector)` waits for a
vector's queued work; this also submits an expression that has not run.

Only the named value runs. Its intermediates stay unrun -- forcing those would
cost a kernel and an MRAM buffer each.

    res = op(da, db)     # nothing has run
    fence(res)           # runs here, so this is where the time is spent
    Array(res)           # already computed
"""
fence(x::DpuLazy) = (fence(DpuVector(x)); x)

const _Lane = Union{DpuVector,DpuLazy}

# An operand slot needs a real vector; anything else passes through.
_force(x) = x
_force(x::DpuLazy) = DpuVector(x)

# Operators on an unrun expression keep it unrun.  On a DpuVector they stay
# eager: `a + b` is a statically compiled kernel.
_lazy(bc::Base.Broadcast.Broadcasted) = DpuLazy(bc, _bclength(bc))

Base.:-(x::DpuLazy) = _lazy(Base.broadcasted(-, x))
Base.abs(x::DpuLazy) = _lazy(Base.broadcasted(abs, x))

for f in (:+, :-, :*, :div, :(>>), :(==), :<, :>, :<=, :>=)
    @eval Base.$f(x::DpuLazy, y::Union{DpuLazy,DpuVector,Integer}) =
        _lazy(Base.broadcasted($f, x, y))
    @eval Base.$f(x::Union{DpuVector,Integer}, y::DpuLazy) =
        _lazy(Base.broadcasted($f, x, y))
end

# A reduction over an unrun expression is one pass: the terminal is appended to
# the program instead of reducing a materialised intermediate.
for (f, terminal) in ((:sum, :sum), (:prod, :prod),
                      (:minimum, :minimum), (:maximum, :maximum))
    @eval function Base.$f(x::DpuLazy)
        e, primary, operands = _lower_tree(x.bc)
        x.consumed = true       # reduced here, so `sync()` must not run it too
        return dpu_pipeline_reduce(primary, $terminal(e); operands = operands)
    end
end

Base.sum(f, x::DpuLazy) = sum(Base.broadcasted(f, x))
Base.prod(f, x::DpuLazy) = prod(Base.broadcasted(f, x))
Base.minimum(f, x::DpuLazy) = minimum(Base.broadcasted(f, x))
Base.maximum(f, x::DpuLazy) = maximum(Base.broadcasted(f, x))

"""
    dest .= expr

Writes through `dest`'s existing buffer, so other handles to it observe the
result. One kernel pass.
"""
Base.copyto!(dest::DpuVector, x::DpuLazy) =
    (x.consumed = true; copyto!(dest, x.bc))

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

export select_op

# ---- RPN pipelines ----
#
# `transform` and `reduce_expr` are the Julia equivalents of the C++
# transform()/reduce() expression lambdas.  The program is built here (see
# expr.jl) and submitted through pipeline()/pipeline_reduce(), so it fuses the
# same way and also works when the library was built with JIT=0.

function _veclist(vs)
    vs = map(_force, vs)
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
function dpu_pipeline(v::_Lane, e::DpuExpr;
                  operands::AbstractVector{DpuVector} = DpuVector[],
                  scalars::AbstractVector{<:Integer} = Int32[])
    _check_program(e, operands)
    sc = Int32.(collect(scalars))
    handle = retry_on_oom(() -> PolymerPIM.launch_pipeline(
        _force(v).handle, e.ops, _veclist(operands), sc))
    return DpuVector(handle)
end

"""
    dpu_pipeline_reduce(v, e; operands, scalars) -> DpuFuture

As [`dpu_pipeline`](@ref), but `e` must end in a reduction terminal (`sum`, `prod`,
`minimum`, `maximum`). Returns a future so independent reductions still fuse.
"""
function dpu_pipeline_reduce(v::_Lane, e::DpuExpr;
                          operands::AbstractVector{DpuVector} = DpuVector[],
                          scalars::AbstractVector{<:Integer} = Int32[])
    _check_program(e, operands)
    isempty(e.ops) && throw(ArgumentError("empty program"))
    Opcodes.is_reduction(e.ops[end]) || throw(ArgumentError(
        "program must end in a reduction terminal (sum/prod/minimum/maximum)"))
    sc = Int32.(collect(scalars))
    handle = retry_on_oom(() -> PolymerPIM.launch_pipeline_reduce(
        _force(v).handle, e.ops, _veclist(operands), sc))
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
function transform(f, v::_Lane, operands::_Lane...;
                   scalars::AbstractVector{<:Integer} = Int32[])
    exprs = DpuExpr[input()]
    for i in 1:length(operands)
        push!(exprs, operand(i))
    end
    return dpu_pipeline(_force(v), f(exprs);
                        operands = DpuVector[map(_force, operands)...],
                        scalars = scalars)
end

"""
    reduce_expr(f, v, operands...; scalars=Int32[]) -> DpuFuture

As [`transform`](@ref), but `f` must return a reduction. Left unread, several of
these fuse into a single kernel pass.

    reduce_expr(a, b) do x
        sum(x[1] * x[2])          # dot product
    end
"""
function reduce_expr(f, v::_Lane, operands::_Lane...;
                     scalars::AbstractVector{<:Integer} = Int32[])
    exprs = DpuExpr[input()]
    for i in 1:length(operands)
        push!(exprs, operand(i))
    end
    return dpu_pipeline_reduce(_force(v), f(exprs);
                               operands = DpuVector[map(_force, operands)...],
                               scalars = scalars)
end

"""
    dpu_pipeline_multi(v, chains; operands, scalars) -> Vector{DpuVector}

Run several independent chains over `v` in **one** kernel pass, one result
vector per chain. The chains see the same `input()`, `operand(i)` and
`scalar_var(i)`, so shared loads happen once.

    values, labels = dpu_pipeline_multi(a, [best, argmax(lanes)];
                                       operands = [b, c])

This is the shape horizontal fusion produces when it merges independent
programs; here it is submitted directly, so it does not depend on the two
programs landing next to each other in the queue. The results are written
through their own buffers, so -- as with `dest .= expr` -- they do not
vertically fuse into a later consumer.
"""
function dpu_pipeline_multi(v::DpuVector, chains::AbstractVector{DpuExpr};
                            operands::AbstractVector{DpuVector} = DpuVector[],
                            scalars::AbstractVector{<:Integer} = Int32[])
    isempty(chains) && throw(ArgumentError("need at least one chain"))
    length(chains) <= MAX_CHAINS || throw(ArgumentError(
        "$(length(chains)) chains exceeds MAX_HFUSE_CHAINS ($MAX_CHAINS)"))
    program = chain(chains...)
    _check_program(program, operands)
    dests = [DpuVector(length(v)) for _ in chains]
    sc = Int32.(collect(scalars))
    retry_on_oom(() -> PolymerPIM.launch_pipeline_multi(
        _veclist(dests), v.handle, program.ops, _veclist(operands), sc))
    return dests
end

export dpu_pipeline, dpu_pipeline_reduce, dpu_pipeline_multi, transform, reduce_expr

# ---- per-element winner across K vectors ----
#
# `argmin(collection)` is Base's index of the smallest element, so the winning
# lane per position is `argmin.(zip(v1, v2, v3))` instead.  1-based, as Julia's
# is over a tuple.

const DpuZip = Base.Iterators.Zip{<:Tuple{_Lane,Vararg{_Lane}}}

# Any other broadcast over a zip would have Base collect it -- one readback
# per element.
Base.broadcastable(::DpuZip) = throw(ArgumentError(
    "only argmin./argmax. are supported over zip(::DpuVector...); another " *
    "broadcast would collect the vectors to the host one element at a time"))

function _lane_program(nlanes::Integer, want_max::Bool)
    lanes = DpuExpr[input(); [operand(j) for j in 1:(nlanes - 1)]]
    label = want_max ? argmax(lanes) : argmin(lanes)
    return lanes, label
end

# Lazy like any other broadcast, so `argmin.(zip(a, b)) .* 3` is one program.
# LaneArg is a marker for the lowering to dispatch on; it is never called.
struct LaneArg{Max} end

Base.broadcasted(::typeof(argmin), z::DpuZip) =
    Base.broadcasted(DpuStyle(), LaneArg{false}(), z.is...)
Base.broadcasted(::typeof(argmax), z::DpuZip) =
    Base.broadcasted(DpuStyle(), LaneArg{true}(), z.is...)

function _find_lanes(vs::AbstractVector{DpuVector}, want_max::Bool)
    isempty(vs) && throw(ArgumentError("need at least one vector"))
    lanes, label = _lane_program(length(vs), want_max)
    value = _best_expr(lanes, want_max)
    if MAX_CHAINS < 2   # room for one chain only; a pass each
        return (dpu_pipeline(vs[1], value; operands = vs[2:end]),
                dpu_pipeline(vs[1], label; operands = vs[2:end]))
    end
    values, labels = dpu_pipeline_multi(vs[1], [value, label];
                                       operands = vs[2:end])
    return values, labels
end

"""
    findmin_lanes(vectors) -> (values, labels)
    findmax_lanes(vectors) -> (values, labels)

Per element, the winning value and the 1-based index of the vector it came
from, in one kernel pass. Julia's `findmin.(zip(v1, v2, v3))` is a vector of
tuples, which a DPU cannot hold, so the columns come back unzipped:

    values, labels = findmin_lanes([v1, v2, v3])
    collect(zip(Array(values), Array(labels))) == findmin.(zip(a1, a2, a3))
"""
findmin_lanes(vs::AbstractVector{DpuVector}) = _find_lanes(vs, false)
findmax_lanes(vs::AbstractVector{DpuVector}) = _find_lanes(vs, true)

export findmin_lanes, findmax_lanes

# No min-of-K-vectors opcode, so the value is a select chain.  Strict
# comparison, so ties keep the lowest lane and value and label agree.
function _best_expr(lanes::AbstractVector{DpuExpr}, want_max::Bool)
    best = lanes[1]
    for j in 2:length(lanes)
        best = select(want_max ? best < lanes[j] : lanes[j] < best, lanes[j], best)
    end
    return best
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

export min_squared_distance

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

# ---- local (WRAM) scatter accumulators ----
#
# `bins[idx] .+= v` records an accumulation; nothing launches.  Index and value
# stay lazy, so several updates lower into one program.  Reading a local, or
# `sync()`, flushes them.

"""
    DpuLocalVector(n; reduce_op = :sum)

A small per-DPU accumulator array in WRAM. Scatter into it by indexing with a
lazy expression, and read it with `Array`, which merges every DPU's copy with
`reduce_op`:

    bins = DpuLocalVector(16)
    bins[(da .* 16) .>> 10] .+= 1        # queued
    Array(bins)                          # flushed, then merged

`reduce_op` is one of `:sum`, `:product`, `:min`, `:max`, and is also the
accumulation each update performs, so `.+=` belongs to a `:sum` local and
`.= min.(...)` to a `:min` one.
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

# Queued updates, in the order written: one flush emits one program, so updates
# to different locals still share a pass.
struct _PendingUpdate
    target::DpuLocalVector
    op::UInt8
    index::Any
    value::Any
end

const _PENDING_UPDATES = _PendingUpdate[]

# `bins[idx]` on its own is not a value -- only the target of an accumulation.
struct _LocalSlot
    target::DpuLocalVector
    index::Any
end

# `bins[i] .+= v` desugars to `bins[i] .= bins[i] .+ v`: only the assignment
# target goes through dotview, so the read needs the same slot object.
Base.dotview(l::DpuLocalVector, index) = _LocalSlot(l, index)
Base.getindex(l::DpuLocalVector, index) = _LocalSlot(l, index)

const _ACCUM_OPS = Dict{Any,UInt8}(
    (+) => Opcodes.OP_SUM, (*) => Opcodes.OP_PRODUCT,
    min => Opcodes.OP_MIN, max => Opcodes.OP_MAX,
)

struct _LocalAccum
    slot::_LocalSlot
    op::UInt8
    value::Any
end

function Base.broadcasted(f, slot::_LocalSlot, value)
    op = get(_ACCUM_OPS, f, nothing)
    op === nothing && throw(ArgumentError(
        "a local accumulator supports .+=, .*=, min and max, not $f"))
    return _LocalAccum(slot, op, value)
end

Base.broadcasted(f, value, slot::_LocalSlot) = Base.broadcasted(f, slot, value)

function Base.materialize!(dest::_LocalSlot, acc::_LocalAccum)
    acc.slot.target === dest.target || throw(ArgumentError(
        "a local accumulation reads and writes the same local vector"))
    acc.op == _local_reduce_opcode(dest.target.reduce_op) || throw(ArgumentError(
        "this accumulation is $(acc.op) but the local vector merges with " *
        ":$(dest.target.reduce_op); they have to agree"))
    push!(_PENDING_UPDATES, _PendingUpdate(dest.target, acc.op, dest.index,
                                           acc.value))
    return dest.target
end

_no_accum() = throw(ArgumentError(
    "a local vector can only be accumulated into: bins[i] .+= v"))
Base.materialize!(::_LocalSlot, ::Any) = _no_accum()
Base.materialize!(::_LocalSlot, ::Base.Broadcast.Broadcasted) = _no_accum()

_lower_operand(x, st::_Lowering) = _lower(x, st)
_lower_operand(x::Base.Broadcast.Broadcasted, st::_Lowering) = _lower(x, st)

# The queued updates as one program, without launching it: what flush_locals!
# submits, and what `@code_jitted` shows.
function _pending_program(updates::Vector{_PendingUpdate};
                          consume::Bool = true)
    st = _Lowering()
    locals = DpuLocalVector[]
    reductions = _LocalReduce[]
    for u in updates
        index = _lower_operand(u.index, st)
        value = _lower_operand(u.value, st)
        slot = findfirst(l -> l === u.target, locals)
        if slot === nothing
            push!(locals, u.target)
            slot = length(locals)
        end
        push!(reductions, _LocalReduce(slot - 1, u.op, index, value))
    end
    length(locals) <= MAX_LOCAL_SCRATCH_VECTORS || throw(ArgumentError(
        "$(length(locals)) local vectors exceeds MAX_LOCAL_SCRATCH_VECTORS " *
        "($MAX_LOCAL_SCRATCH_VECTORS); WRAM has room for no more, and the " *
        "extras would silently read back as zeros"))
    st.primary === nothing && throw(ArgumentError(
        "a scatter needs a DpuVector in its index or value"))
    # Folded in, so `sync()` must not run them a second time.
    consume && for x in st.inlined
        x.consumed = true
    end
    return _scatter_program(reductions), st.primary, st.operands, locals
end

# Everything alive, unrun and not folded into something else: the values a
# caller is still holding.  A step in a larger expression is referenced by its
# consumer, so it is not one of these.
function _dangling_lazies()
    live, keep = DpuLazy[], WeakRef[]
    for wr in _LAZY_REGISTRY
        x = wr.value
        x === nothing && continue
        push!(keep, wr)
        (x.forced === nothing && !x.consumed) && push!(live, x)
    end
    resize!(_LAZY_REGISTRY, 0)
    append!(_LAZY_REGISTRY, keep)
    isempty(live) && return DpuLazy[]

    referenced = Base.IdSet{DpuLazy}()
    for x in live
        _mark_referenced(x.bc, referenced)
    end
    return [x for x in live if !(x in referenced)]
end

function _mark_referenced(bc::Base.Broadcast.Broadcasted, seen)
    for a in bc.args
        _mark_referenced(a, seen)
    end
    return nothing
end
function _mark_referenced(x::DpuLazy, seen)
    x in seen && return nothing
    push!(seen, x)
    _mark_referenced(x.bc, seen)
    return nothing
end
_mark_referenced(::Any, _) = nothing

# Run the values nothing else is going to run.  `sync()`'s half of the bargain:
# a caller who kept a result gets it computed, without the steps behind it
# being materialised.
function _run_dangling_lazies()
    for x in _dangling_lazies()
        DpuVector(x)
    end
    return nothing
end

"""
    flush_locals!()

Launch the scatter program the queued updates describe, if any. Called by
[`sync`](@ref) and by reading a local vector, so it rarely needs calling
directly.
"""
function flush_locals!()
    isempty(_PENDING_UPDATES) && return nothing
    updates = copy(_PENDING_UPDATES)
    empty!(_PENDING_UPDATES)
    program, primary, operands, locals = _pending_program(updates)
    ll = PolymerPIM.DpuLocalList()
    for l in locals
        PolymerPIM.var"locallist_push!"(ll, l.handle)
    end
    retry_on_oom(() -> PolymerPIM.launch_pipeline_scatter(
        primary.handle, program.ops, _veclist(operands), Int32[], ll))
    return nothing
end

function Base.Array(l::DpuLocalVector)
    flush_locals!()
    out = Vector{Int32}(undef, l.len)
    retry_on_oom(() -> PolymerPIM.var"local_to_cpu!"(l.handle, out))
    return out
end

export DpuLocalVector, flush_locals!
