# Low-level RPN builders and launch primitives.

using PolymerPIM: Internal
using PolymerPIM.Internal: DpuExpr, input, operand, constant, scalar_var
using PolymerPIM.Internal: dup, sqr, select, lane_index
using PolymerPIM.Internal: add_var, mul_var, shr_var
using PolymerPIM.Internal: transform, reduce_expr
using PolymerPIM.Internal: dpu_pipeline, dpu_pipeline_reduce, dpu_pipeline_multi

@testset "public boundary" begin
    @test !(:DpuExpr in names(PolymerPIM))
    @test !(:transform in names(PolymerPIM))
    @test !(:reduce_expr in names(PolymerPIM))
    @test !(:dpu_pipeline in names(PolymerPIM))
    @test isdefined(Internal, :DpuExpr)
    @test isdefined(Internal, :dpu_pipeline)
    @test !(:DpuExpr in names(Internal))
    @test !(:dpu_pipeline in names(Internal))
end

@testset "broadcast matches hand-built RPN" begin
    a = DpuVector(Int32.(1:N)); b = DpuVector(Int32.(N:-1:1))
    lazy = @code_jitted abs2.(a .- b)
    xs = DpuExpr[input(), operand(1)]
    byhand = code_jitted(sqr(xs[1] - xs[2]); nelements = length(a), noperands = 1)
    @test lazy.ops == byhand.ops
    @test lazy.hash == byhand.hash
end

@testset "scatter matches hand-built RPN" begin
    depth, nbins = 10, 16
    da = DpuVector(Int32[i % (1 << depth) for i in 0:255])
    bucket = shr_var(mul_var(input(), 1), 2)
    byhand = Internal._scatter_program(
        [Internal._LocalReduce(0, Internal.Opcodes.OP_SUM, bucket,
                               scalar_var(3))])

    bins = DpuLocalVector(nbins)
    bins[(da .* Int32(nbins)) .>> Int32(depth)] .+= 1
    program, primary, operands, scalars, locals = PolymerPIM._pending_program(
        PolymerPIM._PENDING_UPDATES; consume = false)
    queued = length(PolymerPIM._PENDING_UPDATES)
    shown = @code_jitted bins[(da .* Int32(nbins)) .>> Int32(depth)] .+= 1

    @test length(PolymerPIM._PENDING_UPDATES) == queued
    @test shown.ops == byhand.ops
    @test shown.nelements == length(da)
    @test occursin("local_accum_0[", shown.source)
    @test program.ops == byhand.ops
    @test primary === da
    @test isempty(operands)
    @test scalars == Int32[nbins, depth, 1]
    @test length(locals) == 1
    @test code_jitted(program).hash == code_jitted(byhand).hash
    sync()
end

@testset "expression builder" begin
    av = Int32.(collect(1:N))
    bv = Int32.(collect(N:-1:1))
    a = DpuVector(av); b = DpuVector(bv)

    # arithmetic
    @test Array(transform(a, b) do x; x[1] + x[2] end) == av .+ bv
    @test Array(transform(a, b) do x; x[1] - x[2] end) == av .- bv
    @test Array(transform(a, b) do x; x[1] * x[2] end) == av .* bv
    @test Array(transform(a, b) do x; div(x[1], x[2]) end) == div.(av, bv)

    # unary and stack ops
    @test Array(transform(a) do x; -x[1] end) == .-av
    @test Array(transform(a) do x; abs(-x[1]) end) == abs.(av)
    @test Array(transform(a) do x; sqr(x[1] - 3) end) == (av .- 3) .^ 2
    @test Array(transform(a) do x; dup(x[1]) + x[1] end) == av .* 2

    # immediates
    @test Array(transform(a) do x; x[1] * 3 + 1 end) == av .* 3 .+ 1
    @test Array(transform(a) do x; x[1] >> 2 end) == av .>> 2
    @test Array(transform(a) do x; x[1] * constant(2) end) == av .* 2
    @test Array(transform(a) do x; 5 - x[1] end) == 5 .- av

    # select
    @test Array(transform(a, b) do x
        select(x[1] > x[2], x[1], x[2])
    end) == max.(av, bv)
end

@testset "expression comparisons" begin
    av = Int32[3, 1, 4, 1, 5, 9, 2, 6]
    bv = Int32[2, 7, 1, 8, 2, 8, 1, 8]
    a = DpuVector(av); b = DpuVector(bv)

    @test Array(transform(a, b) do x; x[1] < x[2] end)  == Int32.(av .< bv)
    @test Array(transform(a, b) do x; x[1] > x[2] end)  == Int32.(av .> bv)
    @test Array(transform(a, b) do x; x[1] <= x[2] end) == Int32.(av .<= bv)
    @test Array(transform(a, b) do x; x[1] >= x[2] end) == Int32.(av .>= bv)
    @test Array(transform(a, b) do x; x[1] == x[2] end) == Int32.(av .== bv)

    # the DpuVector-level operators, which route through RPN
    @test Array(a > b)  == Int32.(av .> bv)
    @test Array(a >= b) == Int32.(av .>= bv)
    @test Array(a <= b) == Int32.(av .<= bv)
    @test Array(a == b) == Int32.(av .== bv)
    @test Array(a > 3)  == Int32.(av .> 3)
    @test Array(a <= 4) == Int32.(av .<= 4)
    @test Array(a .> b) == Int32.(av .> bv)
end

@testset "runtime scalars" begin
    av = Int32.(collect(1:N))
    a = DpuVector(av)
    # Same program, different scalar slot contents.
    e1 = transform(a; scalars = Int32[10]) do x; add_var(x[1], 1) end
    e2 = transform(a; scalars = Int32[100]) do x; add_var(x[1], 1) end
    @test Array(e1) == av .+ 10
    @test Array(e2) == av .+ 100

    @test Array(transform(a; scalars = Int32[3]) do x
        mul_var(x[1], 1)
    end) == av .* 3
end

@testset "expression reductions" begin
    av = Int32.(collect(1:N))
    bv = Int32.(fill(2, N))
    a = DpuVector(av); b = DpuVector(bv)

    @test get(reduce_expr(a, b) do x; sum(x[1] * x[2]) end) ==
          sum(Int64.(av) .* Int64.(bv))
    @test get(reduce_expr(a) do x; maximum(x[1]) end) == maximum(av)
    @test get(reduce_expr(a) do x; minimum(-x[1]) end) == -maximum(av)
    # The DPU accumulates a sum in Int32 unless the library is built with
    # ENABLE_PROMOTION_REDUCTIONS, so keep this one inside 32-bit range.
    small = Int32.(collect(1:64))
    @test get(reduce_expr(DpuVector(small)) do x; sum(sqr(x[1] - 1)) end) ==
          sum(Int64.(small .- 1) .^ 2)

    # a non-reduction program is rejected rather than silently misread
    @test_throws ArgumentError reduce_expr(a) do x; x[1] + 1 end
end

@testset "reductions still fuse through RPN" begin
    n = 512
    vs = [DpuVector(Int32.(collect(1:n) .+ k)) for k in 1:6]
    PolymerPIM.sync()

    before = PolymerPIM.stat_compute_launches()
    fs = [reduce_expr(vs[i], vs[i + 1]) do x; sum(x[1] * x[2]) end
          for i in 1:5]
    vals = [get(f) for f in fs]
    passes = PolymerPIM.stat_compute_launches() - before

    want = [sum(Int64.(collect(1:n) .+ i) .* Int64.(collect(1:n) .+ i .+ 1))
            for i in 1:5]
    @test vals == want
    # Five independent reductions left unread share one kernel pass.
    @test passes == 1
end

@testset "lane_index is shard-local" begin
    n = 40
    idx = Array(transform(DpuVector(zeros(Int32, n))) do x
        lane_index()
    end)
    # Restarts at 0 on every DPU, so it is repeated ranges, not 0:n-1.
    @test minimum(idx) == 0
    @test all(i -> 0 <= i < n, idx)
    @test length(unique(idx)) < n
end

@testset "program limits are validated" begin
    a = DpuVector(Int32.(collect(1:N)))
    @test_throws ArgumentError operand(0)
    @test_throws ArgumentError operand(MAX_VFUSE_INPUTS + 1)
    @test_throws ArgumentError scalar_var(0)
    @test_throws ArgumentError scalar_var(MAX_PIPELINE_SCALARS + 1)

    # A generated kernel has no opcode-buffer ceiling, so a long chain is
    # simply correct rather than rejected.
    long = input()
    for _ in 1:300
        long = long + 1
    end
    @test Array(dpu_pipeline(a, long)) == Int32.(collect(1:N)) .+ 300
end

@testset "raw pipeline submission" begin
    av = Int32.(collect(1:N))
    a = DpuVector(av)
    # pipeline/pipeline_reduce are the primitives transform/reduce_expr use.
    @test Array(dpu_pipeline(a, -input())) == .-av
    @test get(dpu_pipeline_reduce(a, sum(input()))) == sum(Int64.(av))
end

# Independent chains, one program, one pass -- the shape horizontal fusion
# builds by itself, submitted directly.
@testset "dpu_pipeline_multi" begin
    av = Int32.(1:N); bv = Int32.(N:-1:1)
    a = DpuVector(av); b = DpuVector(bv)

    # MAX_HFUSE_CHAINS is a swept build parameter, so how many chains fit is
    # whatever this library was compiled with.
    programs = [input() + operand(1), input() - operand(1), sqr(input())]
    expected = [av .+ bv, av .- bv, av .* av]
    k = min(MAX_CHAINS, length(programs))

    before = PolymerPIM.stat_compute_launches()
    outs = dpu_pipeline_multi(a, programs[1:k]; operands = [b])
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1

    @test length(outs) == k
    for j in 1:k
        @test Array(outs[j]) == expected[j]
    end

    # Scalars and operands are shared by every chain.
    if MAX_CHAINS >= 2
        outs = dpu_pipeline_multi(a, [add_var(input(), 1), mul_var(input(), 1)];
                                  scalars = Int32[7])
        @test Array(outs[1]) == av .+ 7
        @test Array(outs[2]) == av .* 7
    end

    @test_throws ArgumentError dpu_pipeline_multi(a, DpuExpr[])
    @test_throws ArgumentError dpu_pipeline_multi(a, [input() for _ in 1:(MAX_CHAINS + 1)])
end

# Adjacent immediates on the same value are one instruction, so the 1-based
# `argmin` costs what the raw opcode did.
@testset "immediates fold" begin
    e = input()
    @test (e + 1 - 1).ops == e.ops
    @test (e - 1 + 1).ops == e.ops
    @test (e + 2 + 3).ops == (e + 5).ops
    @test (e + 5 - 2).ops == (e + 3).ops
    @test (e - 2 - 3).ops == (e - 5).ops
    @test (e * 3 * 4).ops == (e * 12).ops
    @test (e * 3 * 0).ops == (e * 0).ops
    @test length((e * 2 * 1).ops) == length((e * 2).ops)

    # Only within a class, and only on the tail.
    @test length(((e + 1) * 2).ops) == length((e + 1).ops) + 5
    @test length((div(e, 2) * 2).ops) == length(div(e, 2).ops) + 5
    @test length((div(div(e, 2), 3)).ops) == length(div(e, 2).ops) + 5

    # An immediate whose bytes contain an opcode value must not be mistaken for
    # one: 0x03 is OP_ADD, and the walk starts from the front, so it isn't.
    @test (e + 0x03 + 1).ops == (e + 4).ops

    # The lane label folds back to the bare opcode.
    lanes = DpuExpr[input(), operand(1)]
    zeroed = argmin(lanes) - 1
    @test zeroed.ops[end - 1] == Internal.Opcodes.OP_ARGMIN_K
    @test zeroed.ops[end] == 0x02

    # ... and still computes what it did before.
    a = DpuVector(Int32.(1:N)); b = DpuVector(Int32.(N:-1:1))
    @test Array(transform(a, b) do x
        argmin([x[1], x[2]]) - 1
    end) == argmin.(zip(Array(a), Array(b))) .- 1
    @test Array(transform(a) do x
        x[1] + 7 - 7
    end) == Array(a)
end
