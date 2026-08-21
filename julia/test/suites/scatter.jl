# Local (WRAM) scatter accumulators: indexing a local vector with a lazy
# expression accumulates into a small per-DPU array instead of producing a value
# per element.  This is what backs histogram- and kmeans-style workloads.

@testset "histogram into one accumulator" begin
    n = 512
    data = Int32[i % 8 for i in 0:(n - 1)]
    a = DPUVector(data)

    bins = DpuLocalVector(8)
    @test length(bins) == 8
    bins[a] .+= 1
    @test Array(bins) == Int32[count(==(b), data) for b in 0:7]
end

@testset "a computed bucket index" begin
    n = 1024
    depth, nbins = 10, 16
    data = Int32[i % (1 << depth) for i in 0:(n - 1)]
    a = DPUVector(data)

    bins = DpuLocalVector(nbins)
    bins[(a .* Int32(nbins)) .>> Int32(depth)] .+= 1

    want = zeros(Int32, nbins)
    for v in data
        want[((Int64(v) * nbins) >> depth) + 1] += 1
    end
    @test Array(bins) == want
end

@testset "runtime scalars reach a scatter launch" begin
    data = Int32[i % 4 for i in 0:255]
    a = DPUVector(data)
    offset = 1
    weight = 2

    # Five int32s need padding to 24 bytes; six already occupy 24.
    for nlocal in (5, 6)
        bins = DpuLocalVector(nlocal)
        bins[a .+ offset] .+= weight

        program, primary, operands, scalars, locals = PolymerPIM._pending_program(
            PolymerPIM._PENDING_UPDATES; consume = false)
        @test primary === a
        @test isempty(operands)
        @test scalars == Int32[1, 2]
        @test length(locals) == 1
        @test !isempty(program.ops)

        want = zeros(Int32, nlocal)
        for x in data
            want[x + 2] += 2
        end
        @test Array(bins) == want
    end
end

@testset "shared scatter indices reuse scalar slots" begin
    a = DPUVector(Int32[i % 4 for i in 0:255])
    # Equal values are deliberate: equality must not collapse distinct leaves.
    params = zeros(Int32, MAX_PIPELINE_SCALARS)

    # Model kmeans' shape: the same parameter-heavy index feeds several local
    # updates.  Re-lowering it must not multiply the launch scalar count.
    shared = a
    for p in params
        shared = shared .+ p
    end
    bins = DpuLocalVector(6)
    bins[shared] .+= a
    bins[shared] .+= a

    _, _, _, scalars, _ = PolymerPIM._pending_program(
        PolymerPIM._PENDING_UPDATES)
    @test length(scalars) == MAX_PIPELINE_SCALARS
    empty!(PolymerPIM._PENDING_UPDATES)
end

@testset "several updates share one pass" begin
    # The kmeans shape: one index expression feeding a count plus a sum per
    # dimension.  The shared prefix is emitted once and re-used with OP_DUP.
    n = 256
    slots = 4
    stride = 3                     # [count, sum(a), sum(b)] per slot
    av = Int32[i % slots for i in 0:(n - 1)]
    bv = Int32[i for i in 0:(n - 1)]
    a = DPUVector(av); b = DPUVector(bv)

    acc = DpuLocalVector(slots * stride)
    base = a .* Int32(stride)
    sync(); before = PolymerPIM.stat_compute_launches()
    acc[base] .+= 1
    acc[base .+ Int32(1)] .+= a
    acc[base .+ Int32(2)] .+= b
    sync()                         # queued until here
    @test PolymerPIM.stat_compute_launches() - before == 1

    want = zeros(Int64, slots * stride)
    for i in 1:n
        s = Int64(av[i]) * stride
        want[s + 1] += 1
        want[s + 2] += Int64(av[i])
        want[s + 3] += Int64(bv[i])
    end
    @test Array(acc) == Int32.(want)
end

@testset "min and max accumulators" begin
    n = 256
    slots = 4
    av = Int32[i % slots for i in 0:(n - 1)]
    bv = Int32[(i * 37) % 1000 for i in 0:(n - 1)]
    a = DPUVector(av); b = DPUVector(bv)

    want_lo = fill(typemax(Int32), slots)
    want_hi = fill(typemin(Int32), slots)
    for i in 1:n
        s = av[i] + 1
        want_lo[s] = min(want_lo[s], bv[i])
        want_hi[s] = max(want_hi[s], bv[i])
    end

    lo = DpuLocalVector(slots; reduce_op = :min)
    lo[a] .= min.(lo[a], b)
    @test Array(lo) == want_lo

    hi = DpuLocalVector(slots; reduce_op = :max)
    hi[a] .= max.(hi[a], b)
    @test Array(hi) == want_hi
end

# WRAM fits MAX_LOCAL_SCRATCH_VECTORS locals, which is 1 on some builds.
MAX_LOCAL_SCRATCH_VECTORS >= 2 && @testset "two locals in one program" begin
    n = 128
    slots = 4
    av = Int32[i % slots for i in 0:(n - 1)]
    a = DPUVector(av)

    counts = DpuLocalVector(slots)
    totals = DpuLocalVector(slots)
    sync(); before = PolymerPIM.stat_compute_launches()
    counts[a] .+= 1
    totals[a] .+= a
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test Array(counts) == Int32[count(==(s), av) for s in 0:(slots - 1)]
    @test Array(totals) == Int32[s * count(==(s), av) for s in 0:(slots - 1)]
end

@testset "scatter argument validation" begin
    @test_throws ArgumentError DpuLocalVector(8; reduce_op = :median)
    @test_throws Exception DpuLocalVector(0)

    a = DPUVector(Int32[1, 2, 3, 4])
    # Nothing queued: flushing is a no-op rather than an error.
    sync()
    @test flush_locals!() === nothing

    # The accumulation has to match how the local merges.
    lo = DpuLocalVector(4; reduce_op = :min)
    @test_throws ArgumentError lo[a] .+= 1
    bins = DpuLocalVector(4)
    @test_throws ArgumentError bins[a] .= a
    @test_throws ArgumentError bins[a] .= div.(bins[a], 2)

    # WRAM fits only MAX_LOCAL_SCRATCH_VECTORS locals, and the extras used to
    # read back as zeros rather than failing.
    too_many = [DpuLocalVector(4) for _ in 1:(MAX_LOCAL_SCRATCH_VECTORS + 1)]
    for l in too_many
        l[a] .+= 1
    end
    @test_throws ArgumentError sync()
    empty!(PolymerPIM._PENDING_UPDATES)
end
