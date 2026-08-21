# Internal RPN expression builder used by lazy lowering.
#
# Slot numbering is 1-based on this side: `operand(1)` is the first vector in
# the `operands` list, `scalar_var(1)` the first entry in `scalars`.

using .Opcodes

"""
    DpuExpr

A partially built RPN program used by `PolymerPIM.Internal`.
"""
struct DpuExpr
    ops::Vector{UInt8}
end

DpuExpr() = DpuExpr(UInt8[])

# little-endian 4-byte immediate, matching the interpreter's manual decode
function _imm32(v::Integer)
    b = reinterpret(UInt32, Int32(v))
    return UInt8[b & 0xFF, (b >> 8) & 0xFF, (b >> 16) & 0xFF, (b >> 24) & 0xFF]
end

_cat(parts...) = DpuExpr(vcat((p.ops for p in parts)...))
_append(e::DpuExpr, op::UInt8) = DpuExpr(vcat(e.ops, op))

# ---- leaves ----

"""
    input()

The vector the pipeline is launched on.
"""
input() = DpuExpr([Opcodes.OP_PUSH_INPUT])

"""
    operand(i)

The `i`-th extra vector (1-based) passed alongside the program.
"""
function operand(i::Integer)
    1 <= i <= MAX_VFUSE_INPUTS ||
        throw(ArgumentError("operand slot $i outside 1:$MAX_VFUSE_INPUTS"))
    return DpuExpr([UInt8(Opcodes.OP_PUSH_OPERAND_0 + (i - 1))])
end

"""
    constant(v)

An immediate baked into the program.
"""
constant(v::Integer) = DpuExpr(vcat(Opcodes.OP_PUSH_SCALAR, _imm32(v)))

mutable struct _DpuScalar
    value::Int32
end

"""
    scalar_var(i)

The `i`-th runtime scalar (1-based). Unlike [`constant`](@ref) this does not
change the program, so varying it reuses the same compiled kernel.
"""
function scalar_var(i::Integer)
    1 <= i <= MAX_PIPELINE_SCALARS ||
        throw(ArgumentError("scalar slot $i outside 1:$MAX_PIPELINE_SCALARS"))
    return DpuExpr(UInt8[Opcodes.OP_PUSH_SCALAR_VAR, UInt8(i - 1)])
end

# ---- expr ⊗ expr ----

for (f, op) in ((:+, :OP_ADD), (:-, :OP_SUB), (:*, :OP_MUL),
                (:div, :OP_DIV), (:>>, :OP_ASR))
    @eval Base.$f(a::DpuExpr, b::DpuExpr) = _append(_cat(a, b), Opcodes.$op)
end

# Comparisons yield 1/0 per lane.  These opcodes are implemented by both the
# interpreter and the JIT codegen even though no C++ dpu_vector operator uses
# them, so RPN is the only way to reach >, >=, <= and elementwise ==.
for (f, op) in ((:(==), :OP_EQ), (:<, :OP_LT), (:>, :OP_GT),
                (:>=, :OP_GE), (:<=, :OP_LE))
    @eval Base.$f(a::DpuExpr, b::DpuExpr) = _append(_cat(a, b), Opcodes.$op)
end

# ---- expr ⊗ scalar ----
#
# These use the immediate-carrying opcodes rather than push-then-combine, which
# keeps the generated kernel one value narrower.

# Where the program's last instruction begins.  Walked from the start, since an
# immediate can hold a byte that reads as an opcode; nothing if the program does
# not decode cleanly.
function _last_op_start(ops::Vector{UInt8})
    i, last, n = 1, nothing, length(ops)
    while i <= n
        last = i
        i += 1 + Int(Opcodes.inline_bytes(ops[i]))
    end
    return i == n + 1 ? last : nothing
end

_imm32_at(ops::Vector{UInt8}, at::Int) = reinterpret(Int32,
    UInt32(ops[at]) | UInt32(ops[at + 1]) << 8 |
    UInt32(ops[at + 2]) << 16 | UInt32(ops[at + 3]) << 24)

# Two immediates applied to the same value are one immediate, so `argmin(d) - 1`
# costs what the bare opcode does.  Both classes are associative under the
# wraparound the DPU does anyway; division and shifts are not, so they are left
# alone.  A net identity drops out entirely.
const _ADDITIVE = (Opcodes.OP_ADD_SCALAR, Opcodes.OP_SUB_SCALAR)

function _fold_immediate(ops::Vector{UInt8}, v::Integer, op::UInt8)
    (op in _ADDITIVE || op == Opcodes.OP_MUL_SCALAR) || return nothing
    at = _last_op_start(ops)
    at === nothing && return nothing
    prev = ops[at]
    at + 4 == length(ops) || return nothing        # must be an immediate op
    head = ops[1:(at - 1)]
    prev_v = _imm32_at(ops, at + 1)
    if op in _ADDITIVE && prev in _ADDITIVE
        net = (prev == Opcodes.OP_SUB_SCALAR ? -prev_v : prev_v) +
              (op == Opcodes.OP_SUB_SCALAR ? -Int32(v) : Int32(v))
        net == 0 && return head
        return net > 0 ? vcat(head, Opcodes.OP_ADD_SCALAR, _imm32(net)) :
                         vcat(head, Opcodes.OP_SUB_SCALAR, _imm32(-net))
    elseif op == Opcodes.OP_MUL_SCALAR && prev == Opcodes.OP_MUL_SCALAR
        net = prev_v * Int32(v)
        net == 1 && return head
        return vcat(head, Opcodes.OP_MUL_SCALAR, _imm32(net))
    end
    return nothing
end

function _scalar_op(a::DpuExpr, v::Integer, op::UInt8)
    folded = _fold_immediate(a.ops, v, op)
    folded === nothing || return DpuExpr(folded)
    return DpuExpr(vcat(a.ops, op, _imm32(v)))
end

for (f, op) in ((:+, :OP_ADD_SCALAR), (:-, :OP_SUB_SCALAR),
                (:*, :OP_MUL_SCALAR), (:div, :OP_DIV_SCALAR),
                (:>>, :OP_ASR_SCALAR), (:(==), :OP_EQ_SCALAR),
                (:<, :OP_LT_SCALAR), (:>, :OP_GT_SCALAR),
                (:>=, :OP_GE_SCALAR), (:<=, :OP_LE_SCALAR))
    @eval Base.$f(a::DpuExpr, v::Integer) = _scalar_op(a, v, Opcodes.$op)
end

Base.:+(v::Integer, a::DpuExpr) = a + v
Base.:*(v::Integer, a::DpuExpr) = a * v
Base.:-(v::Integer, a::DpuExpr) = constant(v) - a

# A runtime scalar slot applied in place, same saving as above.
function _scalar_var_op(a::DpuExpr, i::Integer, op::UInt8)
    return DpuExpr(vcat(a.ops, op, UInt8(i - 1)))
end

for (f, op) in ((:add, :OP_ADD_SCALAR_VAR), (:sub, :OP_SUB_SCALAR_VAR),
                (:mul, :OP_MUL_SCALAR_VAR), (:divide, :OP_DIV_SCALAR_VAR),
                (:shr, :OP_ASR_SCALAR_VAR), (:eq, :OP_EQ_SCALAR_VAR),
                (:lt, :OP_LT_SCALAR_VAR), (:gt, :OP_GT_SCALAR_VAR),
                (:ge, :OP_GE_SCALAR_VAR), (:le, :OP_LE_SCALAR_VAR))
    fname = Symbol(f, :_var)
    @eval $fname(a::DpuExpr, i::Integer) = _scalar_var_op(a, i, Opcodes.$op)
end

# ---- unary / stack ----

Base.:-(a::DpuExpr) = _append(a, Opcodes.OP_NEGATE)
Base.abs(a::DpuExpr) = _append(a, Opcodes.OP_ABS)

"""
    dup(e)

Duplicate the top of stack.
"""
dup(a::DpuExpr) = _append(a, Opcodes.OP_DUP)

"""
    sqr(e)

`e * e`, evaluated once. Cheaper than `e * e`, which would load it twice.
"""
sqr(a::DpuExpr) = _append(dup(a), Opcodes.OP_MUL)

"""
    lane_index()

The element's index *within its own DPU shard*, not its global position: the
kernel computes `blk + i`, which restarts at 0 on every DPU. With `n` spread
over `d` DPUs each shard sees `0:(n÷d - 1)`.
"""
lane_index() = DpuExpr([Opcodes.OP_PUSH_INDEX])

"""
    global_index()

The element's index in the whole vector: [`lane_index`](@ref) plus the shard
base the host passes in the launch args. Reduce over this one when the answer
is a position, as `argmax` does.
"""
global_index() = DpuExpr([Opcodes.OP_PUSH_GLOBAL_INDEX])

"""
    select(cond, then_e, else_e)

Per lane: `cond != 0 ? then_e : else_e`.
"""
select(cond::DpuExpr, then_e::DpuExpr, else_e::DpuExpr) =
    _append(_cat(cond, then_e, else_e), Opcodes.OP_SELECT)

# ---- reduction terminals ----

Base.sum(a::DpuExpr) = _append(a, Opcodes.OP_SUM)
Base.prod(a::DpuExpr) = _append(a, Opcodes.OP_PRODUCT)
Base.minimum(a::DpuExpr) = _append(a, Opcodes.OP_MIN)
Base.maximum(a::DpuExpr) = _append(a, Opcodes.OP_MAX)

function _arg_k(lanes::AbstractVector{DpuExpr}, op::UInt8)
    isempty(lanes) && throw(ArgumentError("need at least one lane"))
    length(lanes) <= 255 || throw(ArgumentError("too many lanes"))
    ops = vcat((l.ops for l in lanes)...)
    # +1 because the opcode counts lanes from 0 and Julia counts from 1.
    return DpuExpr(vcat(ops, op, UInt8(length(lanes)))) + 1
end

"""
    argmin(exprs) / argmax(exprs)

Which of `exprs` is smallest (largest) at each element, 1-based. A `DpuExpr` is
one value per element, so this is Base's `argmin` over that collection of
values:

    argmin([d1, d2, d3])          # nearest of three distances, per element

Also spelled `argmin(zip(d1, d2, d3))`. Over `DPUVector`s the broadcast form
`argmin.(zip(v1, v2, v3))` is the same program.
"""
Base.argmin(lanes::AbstractVector{DpuExpr}) = _arg_k(lanes, Opcodes.OP_ARGMIN_K)
Base.argmax(lanes::AbstractVector{DpuExpr}) = _arg_k(lanes, Opcodes.OP_ARGMAX_K)

const ExprZip = Base.Iterators.Zip{<:Tuple{DpuExpr,Vararg{DpuExpr}}}
Base.argmin(z::ExprZip) = Base.argmin(DpuExpr[z.is...])
Base.argmax(z::ExprZip) = Base.argmax(DpuExpr[z.is...])

# ---- chain separation, for building several results in one pass ----

"""
    chain(exprs...)

Concatenate independent chains into one program so they share a kernel pass.
Only meaningful for the multi-output forms.
"""
function chain(exprs::DpuExpr...)
    isempty(exprs) && return DpuExpr()
    out = UInt8[]
    for (i, e) in enumerate(exprs)
        i > 1 && push!(out, Opcodes.OP_NEXT_CHAIN)
        append!(out, e.ops)
    end
    return DpuExpr(out)
end

# ---- local (WRAM) scatter accumulators ----
#
# (index, value) pairs written into small per-DPU arrays, as
# dpu_pipeline_context::materialize_ops does it: the shared index prefix once,
# re-used with OP_DUP, then each value and OP_APPLY_INDIRECT with its slot and
# reduce opcode.  Built by `flush_locals!`, never by hand.

const LOCAL_REDUCE_OPS = (:sum, :product, :min, :max)

struct _LocalReduce
    slot::Int          # 0-based index into the locals list
    reduce_op::UInt8
    index::DpuExpr
    value::DpuExpr
end

_local_reduce_opcode(op::Symbol) =
    op === :sum     ? Opcodes.OP_SUM :
    op === :product ? Opcodes.OP_PRODUCT :
    op === :min     ? Opcodes.OP_MIN :
    op === :max     ? Opcodes.OP_MAX :
    throw(ArgumentError("unknown local reduce op $op"))

# Length of the longest opcode-aligned prefix common to every index program.
# Splitting mid-opcode would emit a truncated instruction, so the boundary is
# walked with inline_bytes rather than compared byte-for-byte.
function _common_index_prefix(rs::Vector{_LocalReduce})
    isempty(rs) && return 0
    first_ops = rs[1].index.ops
    raw = length(first_ops)
    for r in rs[2:end]
        other = r.index.ops
        raw = min(raw, length(other))
        i = 0
        while i < raw && first_ops[i + 1] == other[i + 1]
            i += 1
        end
        raw = i
    end
    common, i = 0, 0
    while i < length(first_ops)
        nxt = i + 1 + Opcodes.inline_bytes(first_ops[i + 1])
        nxt > raw && break
        common = nxt
        i = nxt
    end
    return common
end

function _scatter_program(rs::Vector{_LocalReduce})
    isempty(rs) && return DpuExpr()
    out = UInt8[]
    common = _common_index_prefix(rs)
    common > 0 && append!(out, rs[1].index.ops[1:common])
    for r in rs
        if common > 0
            push!(out, Opcodes.OP_DUP)
            append!(out, r.index.ops[(common + 1):end])
        else
            append!(out, r.index.ops)
        end
        append!(out, r.value.ops)
        push!(out, Opcodes.OP_APPLY_INDIRECT)
        push!(out, UInt8(r.slot))
        push!(out, r.reduce_op)
    end
    return DpuExpr(out)
end
