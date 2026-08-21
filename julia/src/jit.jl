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

# `source` keeps the shared preamble, so it matches the compiled file byte for
# byte; printing skips to the function.
function _body(c::JittedCode)
    at = findfirst("int k_" * c.hash, c.source)
    return at === nothing ? (c.source, false) : (c.source[first(at):end], true)
end

function Base.show(io::IO, ::MIME"text/plain", c::JittedCode)
    println(io, "JIT kernel k_", c.hash, " -- ", length(c.ops), " opcodes, ",
            c.noperands, " operand", c.noperands == 1 ? "" : "s",
            c.nelements > 0 ? ", $(c.nelements) elements" : "")
    body, elided = _body(c)
    # Checked here, not at construction: the kernel is written on first launch.
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
    e, primary, operands = _lower_tree(bc; consume = false)
    return code_jitted(e; nelements = length(primary), noperands = length(operands))
end

# An unrun expression: the program it would submit.  Inspecting must not count
# as a use, or it would change whether `sync()` runs it.
code_jitted(x::DpuLazy) = code_jitted(x.bc)

"""
    iscompiled(c::JittedCode)

Whether the JIT has written this kernel yet. It appears at `c.path` the first
time a program with this `hash` is launched, and is reused from then on.
"""
iscompiled(c::JittedCode) = isfile(c.path)

# Already materialised: it ran a statically compiled kernel (`a + b`, `sum(a)`).
code_jitted(x) = error("""
    a $(typeof(x)) is materialised, so it came from a statically compiled kernel.
    @code_jitted takes broadcasts, reductions over them, and RPN programs:
        @code_jitted a .+ b
        @code_jitted sum(a .+ b)""")

# `sum(a .+ b)`: the chain with a reduction terminal appended.  Julia materialises
# the broadcast first, so evaluating it costs an extra pass; reduce_expr does not.
const REDUCERS = (:sum, :prod, :minimum, :maximum)

function _code_jitted_reduce(f, bc::Base.Broadcast.Broadcasted)
    e, primary, operands = _lower_tree(bc; consume = false)
    return code_jitted(f(e); nelements = length(primary),
                       noperands = length(operands))
end

_code_jitted_reduce(f, e::DpuExpr) = code_jitted(f(e))
_code_jitted_reduce(f, x::DpuLazy) = _code_jitted_reduce(f, x.bc)
_code_jitted_reduce(f, x) = code_jitted(x)

# `sum(abs, a)` / `mapreduce(abs, +, a)`: trace the function, append the terminal.
_code_jitted_map(terminal, f, v::DpuVector) =
    code_jitted(terminal(_trace(f)); nelements = length(v))

function _code_jitted_mapreduce(f, op, v::DpuVector)
    terminal = get(MAPREDUCE_TERMINALS, op, nothing)
    terminal === nothing && throw(ArgumentError("op must be +, *, min or max"))
    return _code_jitted_map(terminal, f, v)
end

# The arg forms launch as they build, so their result is already materialised;
# rebuild the program the launcher submitted.
const ARG_FORMS = (:argmin, :argmax, :findmin, :findmax)


function _code_jitted_scatter(statement)
    before = length(_PENDING_UPDATES)
    statement()
    length(_PENDING_UPDATES) > before || throw(ArgumentError(
        "that statement queued no local accumulation"))
    queued = _PENDING_UPDATES[(before + 1):end]
    resize!(_PENDING_UPDATES, before)
    program, primary, operands, _ = _pending_program(queued; consume = false)
    return code_jitted(program; nelements = length(primary),
                       noperands = length(operands))
end

# Pass 1 is a statically compiled reduction with no source, so show pass 2: the
# index of the value's first occurrence.
_code_jitted_arg(::Symbol, v::DpuVector) =
    code_jitted(_arg_index_program(); nelements = length(v))

_code_jitted_arg(name::Symbol, x) = error(
    "@code_jitted $name takes a DpuVector or a list of them, got a $(typeof(x))")

# Assignments describe their right-hand side.  Stripped before dispatch, so a
# reduction under one stays visible; nothing is launched, so nothing is assigned.
_rhs(x) = x
_rhs(ex::Expr) = (ex.head === :(=) || ex.head === :.=) ? _rhs(ex.args[2]) : ex

# `a .+ b` parses as a call to `.+`, `f.(x)` as Expr(:., f, tuple); both lower to
# materialize(broadcasted(...)).  `broadcasted` keeps the tree lazy instead.
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

Broadcasts stay lazy, so the whole expression shows as the one kernel it becomes
-- including a reduction over one, `@code_jitted sum(a .+ b)`, which folds the
terminal into the same program. `sum(abs, a)` and `mapreduce(abs, +, a)` work the
same way. The reduction must be written by name: a macro cannot see through a
variable holding `sum`.

`a + b` and `sum(a)` use statically compiled kernels and have no generated
source. `argmin` / `argmax` / `findmin` / `findmax` over a single vector show
their index pass; the value pass is a statically compiled reduction.
`argmin.(zip(a, b))` shows the per-element lane kernel.

An assignment describes its right-hand side: `@code_jitted g = sum(a .+ b)` is
`@code_jitted sum(a .+ b)`.
"""
macro code_jitted(ex)
    # A scatter is a statement: queue it, take it back off, show the program.
    if ex isa Expr && ex.head in (:.=, :(.+=), :(.*=)) && ex.args[1] isa Expr &&
       ex.args[1].head === :ref
        return :(_code_jitted_scatter(() -> $(esc(ex))))
    end
    ex = _rhs(ex)
    if ex isa Expr && ex.head === :call && ex.args[1] isa Symbol
        if length(ex.args) == 2 && ex.args[1] in ARG_FORMS
            return :(_code_jitted_arg($(QuoteNode(ex.args[1])),
                                      $(esc(ex.args[2]))))
        elseif length(ex.args) == 2 && ex.args[1] in REDUCERS
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
