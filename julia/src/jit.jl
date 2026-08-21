# What the JIT will generate for an expression, without compiling or launching it.

"""
    JittedCode

The C kernel one RPN program compiles to: the opcode stream `ops`, its cache key
`hash`, the `source`, the `path` `jit_compile` writes it to, and the shape it was
lowered from (`nelements`, `noperands`).
"""
struct JittedCode
    ops::Vector{UInt8}
    hash::String
    source::String
    path::String
    nelements::Int
    noperands::Int
end

# The includes and externs every kernel repeats verbatim; `source` keeps them so
# it stays byte-identical to the compiled file, printing skips to the function.
function _body(c::JittedCode)
    at = findfirst("int k_" * c.hash, c.source)
    return at === nothing ? (c.source, false) : (c.source[first(at):end], true)
end

function Base.show(io::IO, ::MIME"text/plain", c::JittedCode)
    println(io, "JIT kernel k_", c.hash, " -- ", length(c.ops), " opcodes, ",
            c.noperands, " operand", c.noperands == 1 ? "" : "s",
            c.nelements > 0 ? ", $(c.nelements) elements" : "")
    body, elided = _body(c)
    # Checked here rather than at construction: the kernel gets written the
    # first time something launches this program, which may be after this call.
    println(io, "  ", c.path, iscompiled(c) ? " (compiled)" : " (not compiled yet)",
            elided ? "; preamble elided, full text in .source" : "")
    println(io)
    print(io, body)
end

Base.show(io::IO, c::JittedCode) = print(io, "JittedCode(k_", c.hash, ")")

"""
    code_jitted(x)

The generated kernel for a lazy broadcast, a `DpuExpr`, or an opcode stream.
Usually reached through [`@code_jitted`](@ref).
"""
function code_jitted(ops::AbstractVector{UInt8}; nelements = 0, noperands = 0)
    ops = Vector{UInt8}(ops)
    hash = String(jit_hash(ops))
    return JittedCode(ops, hash, String(jit_source(ops)),
                      joinpath(String(jit_dir()), "k_" * hash * ".c"),
                      Int(nelements), Int(noperands))
end

code_jitted(e::DpuExpr; kwargs...) = code_jitted(e.ops; kwargs...)

function code_jitted(bc::Base.Broadcast.Broadcasted)
    e, primary, operands = _lower_tree(bc)
    return code_jitted(e; nelements = length(primary), noperands = length(operands))
end

"""
    iscompiled(c::JittedCode)

Whether the JIT has written this kernel yet. It appears at `c.path` the first
time a program with this `hash` is launched, and is reused from then on.
"""
iscompiled(c::JittedCode) = isfile(c.path)

# Whatever came out was already materialised, so it ran on a statically compiled
# kernel (`a + b`, `sum(a)`) rather than a generated one.
code_jitted(x) = error("""
    a $(typeof(x)) is already materialised, so it came from a statically compiled
    kernel rather than a generated one. @code_jitted describes broadcasts,
    reductions over them, and explicit RPN programs:
        @code_jitted a .+ b
        @code_jitted sum(a .+ b)""")

# `sum(a .+ b)`: the broadcast's chain with a reduction terminal appended, which
# is the kernel the reduction runs.  Julia materialises the broadcast before sum
# sees it, so evaluating it also costs a pass for the intermediate -- use
# reduce_expr for the single-pass form.
const REDUCERS = (:sum, :prod, :minimum, :maximum)

function _code_jitted_reduce(f, bc::Base.Broadcast.Broadcasted)
    e, primary, operands = _lower_tree(bc)
    return code_jitted(f(e); nelements = length(primary),
                       noperands = length(operands))
end

_code_jitted_reduce(f, e::DpuExpr) = code_jitted(f(e))
_code_jitted_reduce(f, x) = code_jitted(x)

# `sum(abs, a)` / `mapreduce(abs, +, a)`: trace the function, append the terminal.
_code_jitted_map(terminal, f, v::DpuVector) =
    code_jitted(terminal(_trace(f)); nelements = length(v))

function _code_jitted_mapreduce(f, op, v::DpuVector)
    terminal = get(MAPREDUCE_TERMINALS, op, nothing)
    terminal === nothing && throw(ArgumentError("op must be +, *, min or max"))
    return _code_jitted_map(terminal, f, v)
end

# `g = sum(a .+ b)` and `c .= a .+ b` describe the program on the right; the
# destination is not written, since nothing is launched.  Stripped before the
# macro looks at the expression so a reduction stays visible under one.
_rhs(x) = x
_rhs(ex::Expr) = (ex.head === :(=) || ex.head === :.=) ? _rhs(ex.args[2]) : ex

# `a .+ b` parses as a call to `.+`, and `f.(x)` as Expr(:., f, tuple); both
# lower to materialize(broadcasted(...)).  Rewriting them to `broadcasted`
# keeps the tree lazy -- materialising it would launch the kernel.
_bcify(x) = x
function _bcify(ex::Expr)
    if ex.head === :call && ex.args[1] isa Symbol &&
       length(string(ex.args[1])) > 1 && string(ex.args[1])[1] == '.'
        op = Symbol(string(ex.args[1])[2:end])
        return Expr(:call, Base.broadcasted, op, map(_bcify, ex.args[2:end])...)
    elseif ex.head === :. && length(ex.args) == 2 &&
           ex.args[2] isa Expr && ex.args[2].head === :tuple
        return Expr(:call, Base.broadcasted, ex.args[1],
                    map(_bcify, ex.args[2].args)...)
    end
    return ex
end

"""
    @code_jitted expr

The C kernel `expr` would be JIT compiled to, without launching it:

    julia> @code_jitted a .+ b .* c

Broadcasts stay lazy, so the whole expression shows as the one kernel it
becomes -- including a reduction over one, `@code_jitted sum(a .+ b)`, which
folds the terminal into the same program. `sum(abs, a)` and
`mapreduce(abs, +, a)` work the same way. The reduction has to be written by
name; being a macro, this cannot see through a variable holding `sum`.

`a + b` and `sum(a)` use statically compiled kernels and have no generated
source.

An assignment describes its right-hand side -- `@code_jitted g = sum(a .+ b)` is
`@code_jitted sum(a .+ b)`. Nothing is launched, so nothing is assigned.
"""
macro code_jitted(ex)
    ex = _rhs(ex)
    if ex isa Expr && ex.head === :call && ex.args[1] isa Symbol
        if length(ex.args) == 2 && ex.args[1] in REDUCERS
            return :(_code_jitted_reduce($(esc(ex.args[1])),
                                         $(esc(_bcify(ex.args[2])))))
        elseif length(ex.args) == 3 && ex.args[1] in REDUCERS
            return :(_code_jitted_map($(esc(ex.args[1])), $(esc(ex.args[2])),
                                      $(esc(ex.args[3]))))
        elseif length(ex.args) == 4 && ex.args[1] === :mapreduce
            return :(_code_jitted_mapreduce($(esc(ex.args[2])),
                                            $(esc(ex.args[3])),
                                            $(esc(ex.args[4]))))
        end
    end
    return :(code_jitted($(esc(_bcify(ex)))))
end
