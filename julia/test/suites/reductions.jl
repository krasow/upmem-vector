# Reductions, and the lazy form that lets independent ones share a pass.

@testset "reductions" begin
    n = 4096
    a = DpuVector(Int32.(collect(1:n)))

    @test sum(a) == Int64(n) * Int64(n + 1) ÷ 2
    @test minimum(a) == 1
    @test maximum(a) == n

    b = DpuVector(fill(Int32(1), n))
    @test sum(b) == n
    @test prod(b) == 1
    @test minimum(b) == 1
    @test maximum(b) == 1
end

@testset "lazy reductions" begin
    a = DpuVector(Int32.(collect(1:N)))
    b = DpuVector(fill(Int32(2), N))

    fa = lazy_sum(a)
    fb = lazy_sum(b)
    @test get(fa) == Int64(N) * Int64(N + 1) ÷ 2
    @test get(fb) == 2 * N

    @test lazy_minimum(a) isa DpuFuture
    @test get(lazy_maximum(a)) == N
end

@testset "reductions fuse into one kernel pass" begin
    # The point of the lazy API: queue several reductions, read none, and
    # they share a kernel.  Eight vectors is inside MAX_HFUSE_CHAINS.
    vectors = [DpuVector(fill(Int32(i), 1024)) for i in 1:8]
    PolymerPIM.dpu_sync()

    before = PolymerPIM.stat_compute_launches()
    totals = sums(vectors)
    after = PolymerPIM.stat_compute_launches()

    @test totals == [1024 * i for i in 1:8]
    # Without fusion this would be 8 passes.
    @test after - before <= 2
end

# `sum(f, v)` takes f as an argument, so unlike `sum(f.(v))` there is no
# intermediate to materialise: one program, one pass.
@testset "mapped reductions are one pass" begin
    a = DpuVector(Int32.(-N÷2:N÷2-1))
    host = Array(a)

    @test sum(abs, a) == sum(abs, host)
    @test sum(x -> x * 2, a) == sum(x -> x * 2, host)
    @test maximum(abs, a) == maximum(abs, host)
    @test minimum(x -> -abs(x), a) == minimum(x -> -abs(x), host)
    @test mapreduce(abs, +, a) == mapreduce(abs, +, host)
    @test mapreduce(abs, max, a) == mapreduce(abs, max, host)

    before = PolymerPIM.stat_compute_launches()
    sum(abs, a)
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1

    @test_throws ArgumentError mapreduce(abs, -, a)
    @test_throws MethodError sum(sqrt, a)
end
