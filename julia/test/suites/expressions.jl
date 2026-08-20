# The RPN expression builder: leaves, operators, terminals, runtime
# scalars, and the raw pipeline primitives underneath transform/reduce_expr.

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
