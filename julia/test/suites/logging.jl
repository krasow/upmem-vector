# @show_log / withlog: the host runtime's log, scoped to a block.

@testset "capture window" begin
    a = DpuVector(Int32.(collect(1:N)))
    b = DpuVector(fill(Int32(2), N))
    c = DpuVector(zeros(Int32, N))

    val, log = withlog(level = 1) do
        c .= a .+ b
    end
    @test val === c
    @test !isempty(log)
    # A launch is the one thing a block that runs a kernel has to report.
    @test occursin("launch=", log)
    @test occursin("[TASK]", log)
    @test Array(c) == Int32.(collect(1:N)) .+ Int32(2)

    # Nothing enqueued, nothing logged: the window is the block, not the run.
    _, quiet = withlog(level = 1) do
        1 + 1
    end
    @test !occursin("launch=", quiet)
end

@testset "level raises detail for the block alone" begin
    a = DpuVector(Int32.(collect(1:N)))
    b = DpuVector(fill(Int32(3), N))
    c = DpuVector(zeros(Int32, N))

    _, terse = withlog(level = 1) do
        c .= a .* b
    end
    _, verbose = withlog(level = 3) do
        c .= a .* b
    end

    @test length(verbose) > length(terse)
    # The per-DPU launch argument table is level 3 only.
    @test occursin("NUM_ELEMS", verbose)
    @test !occursin("NUM_ELEMS", terse)

    # Raising it for one block leaves the process level where it was, so the
    # next block is terse again.
    _, after = withlog(level = 1) do
        c .= a .* b
    end
    @test !occursin("NUM_ELEMS", after)
end

@testset "level is bounded by the compiled ceiling" begin
    ceiling = parse(Int, configuration()["ENABLE_DPU_LOGGING"])
    @test ceiling >= 1   # the package requires it; see __init__

    @test_throws ArgumentError withlog(() -> nothing, level = 0)
    @test_throws ErrorException withlog(() -> nothing, level = ceiling + 1)
    # Rejected before anything is captured, so the stack stays balanced.
    @test PolymerPIM.log_capture_depth() == 0
end

@testset "captures nest and unwind" begin
    a = DpuVector(Int32.(collect(1:N)))
    b = DpuVector(fill(Int32(4), N))
    c = DpuVector(zeros(Int32, N))

    inner_log = ""
    _, outer = withlog(level = 1) do
        _, inner_log = withlog(level = 1) do
            c .= a .+ b
        end
        nothing
    end
    @test occursin("launch=", inner_log)
    # The innermost capture takes the lines of its own scope.
    @test !occursin("launch=", outer)
    @test PolymerPIM.log_capture_depth() == 0

    # A throwing block still pops its frame -- otherwise every later line in
    # the session would disappear into an abandoned buffer.
    @test_throws ErrorException withlog(() -> error("boom"), level = 1)
    @test PolymerPIM.log_capture_depth() == 0
end

@testset "macro form" begin
    a = DpuVector(Int32.(collect(1:N)))
    b = DpuVector(fill(Int32(5), N))
    c = DpuVector(zeros(Int32, N))

    # Prints the log; the value is the block's.
    @test (@show_log level=1 c .= a .+ b) === c
    @test Array(c) == Int32.(collect(1:N)) .+ Int32(5)

    total = @show_log level=2 begin
        c .= a .* b
        sum(c)
    end
    @test total[] == sum(Int64.(collect(1:N)) .* 5)

    @test (@show_log level=1 fence=false c .= a .+ b) === c

    @test_throws LoadError @eval @show_log nonsense=1 c .= a .+ b
end
