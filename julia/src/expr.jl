# RPN expression builder.
#
# The C++ side exposes transform()/reduce() taking an expression lambda, but
# CxxWrap cannot pass a Julia closure into them.  Instead we build the opcode
# stream here and submit it through pipeline()/pipeline_reduce(), which reaches
# the same fused kernel.
#
# Slot numbering is 1-based on this side: `operand(1)` is the first vector in
# the `operands` list, `scalar_var(1)` the first entry in `scalars`.

using .Opcodes

"""
    DpuExpr

A partially built RPN program. Combine with the usual arithmetic and comparison
operators, then hand to [`transform`](@ref) or [`reduce_expr`](@ref).
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

function _scalar_op(a::DpuExpr, v::Integer, op::UInt8)
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

"""
    argmin_lanes(lanes) / argmax_lanes(lanes)

Index (0-based, as the kernel produces it) of the winning lane per element.
"""
function _arg_k(lanes::AbstractVector{DpuExpr}, op::UInt8)
    isempty(lanes) && throw(ArgumentError("need at least one lane"))
    length(lanes) <= 255 || throw(ArgumentError("too many lanes"))
    ops = vcat((l.ops for l in lanes)...)
    return DpuExpr(vcat(ops, op, UInt8(length(lanes))))
end

argmin_lanes(lanes::AbstractVector{DpuExpr}) = _arg_k(lanes, Opcodes.OP_ARGMIN_K)
argmax_lanes(lanes::AbstractVector{DpuExpr}) = _arg_k(lanes, Opcodes.OP_ARGMAX_K)

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

export DpuExpr, input, operand, constant, scalar_var, dup, sqr, select,
       lane_index, argmin_lanes, argmax_lanes, chain
export add_var, sub_var, mul_var, divide_var, shr_var,
       eq_var, lt_var, gt_var, ge_var, le_var
