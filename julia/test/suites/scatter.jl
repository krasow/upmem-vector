# Local (WRAM) scatter accumulators: an RPN program indexes into a small
# per-DPU array and accumulates, rather than producing a value per lane.  This
# is what backs histogram- and kmeans-style workloads.

@testset "histogram into one accumulator" begin
    n = 512
    data = Int32[i % 8 for i in 0:(n - 1)]
    a = DpuVector(data)

    bins = DpuLocalVector(8)
    @test length(bins) == 8
    scatter!([bins], a,
             scatter_program([LocalReduce(0, PolymerPIM.Opcodes.OP_SUM,
                                          input(), constant(Int32(1)))]))
    @test Array(bins) == Int32[count(==(b), data) for b in 0:7]
end

@testset "a computed bucket index" begin
    n = 1024
    depth, nbins = 10, 16
    data = Int32[i % (1 << depth) for i in 0:(n - 1)]
    a = DpuVector(data)

    bins = DpuLocalVector(nbins)
    bucket = (input() * Int32(nbins)) >> Int32(depth)
    scatter!([bins], a,
             scatter_program([LocalReduce(0, PolymerPIM.Opcodes.OP_SUM,
                                          bucket, constant(Int32(1)))]))

    want = zeros(Int32, nbins)
    for v in data
        want[((Int64(v) * nbins) >> depth) + 1] += 1
    end
    @test Array(bins) == want
end

@testset "several reductions sharing an index prefix" begin
    # The kmeans shape: one index expression feeding a count plus a sum per
    # dimension.  The shared prefix is emitted once and re-used with OP_DUP, so
    # this is the case that exercises _common_index_prefix.
    n = 256
    slots = 4
    stride = 3                     # [count, sum(a), sum(b)] per slot
    av = Int32[i % slots for i in 0:(n - 1)]
    bv = Int32[i for i in 0:(n - 1)]
    a = DpuVector(av); b = DpuVector(bv)

    acc = DpuLocalVector(slots * stride)
    base = input() * Int32(stride)
    prog = scatter_program([
        LocalReduce(0, PolymerPIM.Opcodes.OP_SUM, base, constant(Int32(1))),
        LocalReduce(0, PolymerPIM.Opcodes.OP_SUM, base + Int32(1), input()),
        LocalReduce(0, PolymerPIM.Opcodes.OP_SUM, base + Int32(2), operand(1)),
    ])
    scatter!([acc], a, prog; operands = [b])

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
    # One local per program: WRAM fits MAX_LOCAL_SCRATCH_VECTORS of them.
    n = 256
    slots = 4
    av = Int32[i % slots for i in 0:(n - 1)]
    bv = Int32[(i * 37) % 1000 for i in 0:(n - 1)]
    a = DpuVector(av); b = DpuVector(bv)

    want_lo = fill(typemax(Int32), slots)
    want_hi = fill(typemin(Int32), slots)
    for i in 1:n
        s = av[i] + 1
        want_lo[s] = min(want_lo[s], bv[i])
        want_hi[s] = max(want_hi[s], bv[i])
    end

    lo = DpuLocalVector(slots; reduce_op = :min)
    scatter!([lo], a,
             scatter_program([LocalReduce(0, PolymerPIM.Opcodes.OP_MIN,
                                         input(), operand(1))]);
             operands = [b])
    @test Array(lo) == want_lo

    hi = DpuLocalVector(slots; reduce_op = :max)
    scatter!([hi], a,
             scatter_program([LocalReduce(0, PolymerPIM.Opcodes.OP_MAX,
                                         input(), operand(1))]);
             operands = [b])
    @test Array(hi) == want_hi
end

@testset "scatter argument validation" begin
    @test_throws ArgumentError DpuLocalVector(8; reduce_op = :median)
    @test_throws Exception DpuLocalVector(0)
    # An empty program is a no-op rather than an error.
    a = DpuVector(Int32[1, 2, 3, 4])
    bins = DpuLocalVector(4)
    @test scatter!([bins], a, DpuExpr())[1] === bins

    # WRAM fits only MAX_LOCAL_SCRATCH_VECTORS locals, and the extras used to
    # read back as zeros rather than failing.
    prog = scatter_program([LocalReduce(0, PolymerPIM.Opcodes.OP_SUM,
                                       input(), constant(Int32(1)))])
    too_many = [DpuLocalVector(4) for _ in 1:(MAX_LOCAL_SCRATCH_VECTORS + 1)]
    @test_throws ArgumentError scatter!(too_many, a, prog)
end
