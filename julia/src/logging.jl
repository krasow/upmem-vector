# Capturing the host runtime's log for one block of code.

"""
    withlog(f; level = 1, fence = true) -> (value, log)

Run `f`, returning its value and everything the runtime logged while it ran --
the functional form of [`@show_log`](@ref), for when the log is data:

    val, log = withlog(level = 3) do
        c .= a .+ b
    end
    @test count(contains("launch=fused_pipeline"), split(log, '\\n')) == 1

`level` is the detail for this block alone, whatever `VECTORDPU_LOG_LEVEL` is:
1 for launches, fusions and materialised allocations, 2 adds queue waits and
lazy allocations, 3 adds the per-DPU launch argument tables. Asking for more
than the library's `ENABLE_DPU_LOGGING` is an error, not a shorter log.

`fence` syncs on entry and before the log is taken, which is what lines the
window up with the block: a kernel logs when it launches, and without the
trailing fence that is usually after the block returns. The catch is that the
fences are part of the program being logged -- they cut fusion and JIT batching
at the block boundary. Pass `fence = false` to see the stream as it falls out
of the queue instead.

If `f` throws, the log so far goes to `stderr`. Timestamps are milliseconds
since the runtime initialized, not since the block started.
"""
function withlog(f; level::Integer = 1, fence::Bool = true)
    requested = _capture_level(level)
    fence && sync()
    log_capture_begin(requested)
    local value
    try
        value = f()
        fence && sync()   # a kernel logs when it launches, not when it is queued
    catch
        text = String(log_capture_end())
        isempty(text) || print(stderr, text)
        rethrow()
    end
    return value, String(log_capture_end())
end

"""
    @show_log [level=1] [fence=true] expr

Print what the runtime logs while `expr` runs, and return `expr`'s value:

    julia> @show_log level=3 c .= a .+ b .* 2

    julia> @show_log level=2 begin
               c .= a .+ b
               sum(c)
           end

The log is captured rather than streamed, so nothing else on stdout is caught up
in it and `VECTORDPU_LOG_LEVEL` stays where it is. See [`withlog`](@ref) for
`level` and `fence`, and for the form that returns the log instead of printing
it. Each line carries its category, so narrowing is a filter on the text:

    _, log = withlog(() -> (c .= a .+ b), level = 3)
    print(join(filter(contains("[FUSION]"), split(log, '\\n')), "\\n"))
"""
macro show_log(args...)
    isempty(args) && throw(ArgumentError(
        "@show_log needs an expression: @show_log level=3 c .= a .+ b"))
    kws = Expr[]
    for opt in args[1:(end - 1)]
        (opt isa Expr && opt.head === :(=) && opt.args[1] isa Symbol) ||
            throw(ArgumentError("@show_log options are keywords: level=3, fence=false"))
        opt.args[1] in (:level, :fence) ||
            throw(ArgumentError("unknown @show_log option `$(opt.args[1])`; " *
                                "expected `level` or `fence`"))
        push!(kws, Expr(:kw, opt.args[1], esc(opt.args[2])))
    end
    return :(_show_log(() -> $(esc(args[end])); $(kws...)))
end

function _show_log(f; level::Integer = 1, fence::Bool = true)
    value, log = withlog(f; level = level, fence = fence)
    if isempty(log)
        # Otherwise an empty block and a mis-set level look like a broken macro.
        println("@show_log: nothing logged at level ", level)
    else
        print(log)
        endswith(log, '\n') || println()
    end
    return value
end

# Levels above what was compiled in have no call sites to fire.
function _capture_level(level::Integer)
    level >= 1 || throw(ArgumentError("log level must be at least 1, got $level"))
    ceiling = Int(log_max_level())
    if level > ceiling
        error("""
              level $level needs libvectordpu compiled with ENABLE_DPU_LOGGING >= $level \
              (this one has $ceiling); its level $level call sites are #if'd out.

              Rebuild the library and the wrapper against it:
                make LOGGING=$level PIPELINE=1 JIT=1 BACKEND=hw install
                make -C julia clean build""")
    end
    return Int(level)
end
