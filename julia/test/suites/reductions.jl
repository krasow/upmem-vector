# Reductions return futures, so independent ones share a pass by default.

@testset "reductions" begin
    n = 4096
    a = DPUVector(Int32.(collect(1:n)))

    @test sum(a)[] == Int64(n) * Int64(n + 1) ÷ 2
    @test minimum(a)[] == 1
    @test maximum(a)[] == n

    b = DPUVector(fill(Int32(1), n))
    @test sum(b)[] == n
    @test prod(b)[] == 1
    @test minimum(b)[] == 1
    @test maximum(b)[] == 1
end

@testset "a reduction is a future" begin
    a = DPUVector(Int32.(collect(1:N)))
    b = DPUVector(fill(Int32(2), N))

    fa = sum(a)
    @test fa isa DpuFuture
    @test fa[] == Int64(N) * Int64(N + 1) ÷ 2
    # Three spellings of the same read.
    @test get(sum(b)) == 2 * N
    @test fetch(sum(b)) == 2 * N
    @test sum(b)[] == 2 * N
end

@testset "reductions fuse into one kernel pass" begin
    # Read none until all are queued.  Eight is inside MAX_HFUSE_CHAINS.
    vectors = [DPUVector(fill(Int32(i), 1024)) for i in 1:8]
    PolymerPIM.dpu_sync()

    before = PolymerPIM.stat_compute_launches()
    futures = [sum(v) for v in vectors]     # queued, none read
    totals = [f[] for f in futures]
    after = PolymerPIM.stat_compute_launches()

    @test totals == [1024 * i for i in 1:8]
    # Without fusion this would be 8 passes.
    @test after - before <= 2
end

# `sum(f, v)` takes f as an argument, so unlike `sum(f.(v))` there is no
# intermediate to materialise: one program, one pass.
@testset "mapped reductions are one pass" begin
    a = DPUVector(Int32.(-N÷2:N÷2-1))
    host = Array(a)

    @test sum(abs, a)[] == sum(abs, host)
    @test sum(x -> x * 2, a)[] == sum(x -> x * 2, host)
    @test maximum(abs, a)[] == maximum(abs, host)
    @test minimum(x -> -abs(x), a)[] == minimum(x -> -abs(x), host)
    @test mapreduce(abs, +, a)[] == mapreduce(abs, +, host)
    @test mapreduce(abs, max, a)[] == mapreduce(abs, max, host)

    before = PolymerPIM.stat_compute_launches()
    sum(abs, a)
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1

    @test_throws ArgumentError mapreduce(abs, -, a)
    @test_throws MethodError sum(sqrt, a)
end

# Two DPU passes, the value then its first index: has to match Base, ties too.
@testset "index reductions match Base" begin
    a = DPUVector(Int32.([4, 7, 1, 7, -3, 0]))
    host = Array(a)

    @test findmax(a) == findmax(host)
    @test findmin(a) == findmin(host)
    @test argmax(a) == argmax(host) == 2   # first of the two 7s
    @test argmin(a) == argmin(host) == 5

    # Same number, same type, whichever spelling asked for it.
    @test findmax(a)[1] === maximum(a)[]
    @test findmin(a)[1] === minimum(a)[]

    b = DPUVector(Int32.(1:N))
    @test argmax(b) == N
    @test argmin(b) == 1

    @test_throws ArgumentError findmax(DPUVector(0))
    @test_throws ArgumentError argmin(DPUVector(0))

    # The winner wherever it sits: last shard, first, and mid-way through a
    # length that shards raggedly.
    up = DPUVector(Int32.(1:1000))
    @test argmax(up) == 1000 && argmin(up) == 1
    ragged = Int32.(vcat(1:500, 9, 501:998))
    @test argmax(DPUVector(ragged)) == argmax(ragged)
    @test findmin(DPUVector(ragged)) == (Int64(1), 1)

    # A pass for the value and a pass for the index.
    sync(); before = PolymerPIM.stat_compute_launches()
    argmax(a); sync()
    @test PolymerPIM.stat_compute_launches() - before == 2

    # Runtime scalars, so length does not change the program: one kernel.
    @test (@code_jitted argmax(a)).hash == (@code_jitted argmax(up)).hash
    @test PolymerPIM.Internal.Opcodes.OP_PUSH_GLOBAL_INDEX in (@code_jitted argmax(a)).ops
end

# An unrun expression reduces in one pass: the terminal joins the program rather
# than reducing a materialised intermediate.
@testset "reducing a lazy expression is one pass" begin
    a = DPUVector(Int32.(1:N)); b = DPUVector(fill(Int32(3), N))
    av = Array(a); bv = Array(b)

    @test (a .+ b) isa DpuLazy
    sync(); before = PolymerPIM.stat_compute_launches()
    total = sum(a .+ b)[]
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test total == sum(Int64.(av) .+ Int64.(bv))

    @test maximum(a .* b)[] == maximum(Int64.(av) .* Int64.(bv))
    @test minimum(-(a .+ b))[] == -maximum(Int64.(av) .+ Int64.(bv))

    # Forcing without a transfer, then reducing, is two passes.
    sync(); before = PolymerPIM.stat_compute_launches()
    kept = DPUVector(a .+ b)
    @test sum(kept)[] == total
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 2
end

# Forcing the same expression twice must not run it twice.
@testset "forcing is memoised" begin
    a = DPUVector(Int32.(1:N)); b = DPUVector(fill(Int32(2), N))

    sync(); before = PolymerPIM.stat_compute_launches()
    x = a .+ b
    first = Array(x)
    second = Array(x)
    kept = DPUVector(x)
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test first == second
    @test Array(kept) == first
    # Once run, it is that vector wherever it appears.
    @test sum(x .* 0 .+ 1)[] == N
end

# `fence` on an unrun expression: runs that value, and only that value.
@testset "fencing an expression" begin
    a = DPUVector(Int32.(1:N)); b = DPUVector(fill(Int32(2), N))

    sync(); before = PolymerPIM.stat_compute_launches()
    step1 = a .+ b            # an intermediate, never wanted on its own
    res = step1 .* Int32(3)
    fence(res)
    sync()
    # One kernel: the intermediate was inlined, not materialised.
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test res isa DpuLazy
    @test Array(res) == (Array(a) .+ Array(b)) .* 3

    # Already run, so reading it launches nothing more.
    before = PolymerPIM.stat_compute_launches()
    Array(res); sync()
    @test PolymerPIM.stat_compute_launches() - before == 0
end

# `sync()` runs the values nothing else will, and leaves the steps behind them
# alone -- so user code needs one barrier, not two.
@testset "sync runs what nothing else will" begin
    a = DPUVector(Int32.(1:N)); b = DPUVector(fill(Int32(2), N))
    av = Array(a); bv = Array(b)

    # A kept result runs; reading it afterwards costs nothing.
    sync(); before = PolymerPIM.stat_compute_launches()
    res = a .+ b .* Int32(3)
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    before = PolymerPIM.stat_compute_launches()
    @test Array(res) == av .+ bv .* 3
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 0

    # Steps consumed by a later expression are not run on their own.
    sync(); before = PolymerPIM.stat_compute_launches()
    step = a .+ b
    final = step .* Int32(2)
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test Array(final) == (av .+ bv) .* 2

    # Nor is an index folded into a scatter.
    bins = DPULocalVector(8)
    sync(); before = PolymerPIM.stat_compute_launches()
    bins[a .* Int32(0)] .+= 1
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test Array(bins)[1] == N
end

# A reduction over an expression that has already run must reduce the result,
# not re-derive the program.
@testset "reducing an already-run expression" begin
    a = DPUVector(Int32.(1:N)); b = DPUVector(fill(Int32(2), N))
    av = Array(a); bv = Array(b)

    x = a .+ b
    sync()                    # dangling, so this runs it
    @test x.forced !== nothing

    before = PolymerPIM.stat_compute_launches()
    total = sum(x)[]
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test total == sum(Int64.(av) .+ Int64.(bv))
    @test minimum(x)[] == minimum(Int64.(av) .+ Int64.(bv))
end
