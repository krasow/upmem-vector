"""
    @dputime expr

Run `expr` once and print its synchronized wall time. Pending work is drained
before timing, and `sync()` after `expr` keeps lazy DPU execution inside the
measured region. The expression's value is returned.

The first call includes JIT compilation; later calls reuse compiled kernels.

    result = @dputime a .+ b
    @dputime begin
        x = sum(a .* b)
        y = maximum(a)
    end
"""
macro dputime(ex)
    return quote
        PolymerPIM.sync()
        local start = Base.time_ns()
        local value = Base.@__tryfinally($(esc(ex)), PolymerPIM.sync())
        local elapsed = Base.time_ns() - start
        println("DPU time: ", round(elapsed / 1.0e6; digits = 3), " ms")
        value
    end
end
