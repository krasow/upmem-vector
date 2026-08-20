using Test
using UpmemVector

# DPU vectors need at least num_dpus elements to work correctly.
# With NR_DPUS=64 (one full rank), use vectors of 64+ elements.
# For reductions, use even larger vectors to span all tasklets.

const N = 4096  # safe vector length for all DPU counts/tasklets

@testset "UpmemVector" begin

    @testset "construction and conversion" begin
        data = Int32.(collect(1:N))
        v = DpuVector(data)
        @test length(v) == N
        @test size(v) == (N,)
        @test eltype(v) == Int32

        back = Array(v)
        @test back == data
        @test Vector(v) == data
        @test collect(v) == data

        data2 = Int32.(collect(1:2N))
        v2 = DpuVector(data2)
        @test length(v2) == 2N
        @test Array(v2) == data2
    end

    @testset "scalar indexing" begin
        data = Int32.(collect(10:10:10N))
        v = DpuVector(data)
        @test v[1] == Int32(10)
        @test v[N] == Int32(10N)
        @test_throws BoundsError v[0]
        @test_throws BoundsError v[N+1]
    end

    @testset "binary vector-vector" begin
        a_data = Int32.(collect(1:N))
        b_data = Int32.(collect(N:-1:1))
        a = DpuVector(a_data)
        b = DpuVector(b_data)

        @test Array(a + b) == a_data .+ b_data
        @test Array(a - b) == a_data .- b_data
        @test Array(a * b) == a_data .* b_data
        @test Array(div(a, b)) == a_data .÷ b_data
    end

    @testset "binary vector-scalar" begin
        a_data = Int32.(collect(2:2:2N))
        a = DpuVector(a_data)

        @test Array(a + 10)    == a_data .+ Int32(10)
        @test Array(10 + a)    == a_data .+ Int32(10)
        @test Array(a - 1)     == a_data .- Int32(1)
        @test Array(a * 3)     == a_data .* Int32(3)
        @test Array(3 * a)     == a_data .* Int32(3)
        @test Array(div(a, 2)) == a_data .÷ Int32(2)
        @test Array(a >> 1)    == a_data .>> Int32(1)
    end

    @testset "unary operations" begin
        a_data = Int32.(vcat(collect(-N÷2:-1), collect(1:N÷2)))
        a = DpuVector(a_data)

        @test Array(-a)    == -a_data
        @test Array(abs(a)) == abs.(a_data)
    end

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

    @testset "chained operations" begin
        a_data = Int32.(collect(1:N))
        b_data = Int32.(collect(N:-1:1))
        a = DpuVector(a_data)
        b = DpuVector(b_data)

        result = abs(-((a + b) - a))
        @test Array(result) == abs.(-(((a_data .+ b_data) .- a_data)))
    end

    @testset "comparisons and select" begin
        a_data = Int32.(collect(1:N))
        b_data = Int32.(collect(N:-1:1))
        a = DpuVector(a_data)
        b = DpuVector(b_data)

        @test Array(a < b) == Int32.(a_data .< b_data)
        @test Array(a == Int32(7)) == Int32.(a_data .== 7)

        # select(cond, then, else), elementwise on the mask.
        cond = DpuVector(Int32.(a_data .< b_data))
        picked = select_op(cond, a, b)
        @test Array(picked) == ifelse.(a_data .< b_data, a_data, b_data)
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
        UpmemVector.dpu_sync()

        before = UpmemVector.stat_compute_launches()
        totals = sums(vectors)
        after = UpmemVector.stat_compute_launches()

        @test totals == [1024 * i for i in 1:8]
        # Without fusion this would be 8 passes.
        @test after - before <= 2
    end

    @testset "broadcasting" begin
        a_data = Int32.(collect(1:N))
        b_data = fill(Int32(3), N)
        a = DpuVector(a_data)
        b = DpuVector(b_data)

        @test Array(a .+ b)   == a_data .+ b_data
        @test Array(a .- b)   == a_data .- b_data
        @test Array(a .* b)   == a_data .* b_data
        @test Array(a .* 2)   == a_data .* Int32(2)
        @test Array(2 .* a)   == a_data .* Int32(2)
        @test Array(.-a)      == -a_data
        @test Array(abs.(.-a)) == a_data
        @test Array(a .>> 1)  == a_data .>> Int32(1)
        @test Array(a .< b)   == Int32.(a_data .< b_data)
    end

    @testset "in-place operations" begin
        # Chaining these used to double-apply the earlier op, and five in a row
        # deadlocked outright.
        acc = DpuVector(fill(Int32(40), N))
        add!(acc, 10)
        sub!(acc, 3)
        mul!(acc, 4)
        div!(acc, 2)
        shr!(acc, 1)
        @test Array(acc) == fill(Int32(47), N)

        v = DpuVector(fill(Int32(5), N))
        w = DpuVector(fill(Int32(2), N))
        @test Array(add!(v, w)) == fill(Int32(7), N)
        @test Array(mul!(v, w)) == fill(Int32(14), N)
        @test Array(sub!(v, w)) == fill(Int32(12), N)
        @test Array(div!(v, w)) == fill(Int32(6), N)

        # An in-place op returns the same vector, not a copy.
        x = DpuVector(fill(Int32(1), N))
        @test add!(x, 1) === x

        # No in-place form exists for a comparison; the wrapper rejects the
        # opcode rather than silently picking a different one.
        @test_throws Exception apply!(x, 1, UpmemVector.Opcodes.OP_EQ_SCALAR)
    end

    @testset "ragged lengths" begin
        # Any length is safe now.  These used to come back corrupt: the host
        # readback pushed align8(shard) into unpadded slots, so a length that
        # was not a multiple of 2*num_dpus silently lost lanes.
        for n in (1, 2, 7, 15, 17, 33, 100, 1000, 4099, 9973)
            data = Int32.(collect(1:n))
            v = DpuVector(data)
            @test length(v) == n
            back = Array(v)
            @test length(back) == n
            @test back == data

            w = DpuVector(fill(Int32(3), n))
            @test Array(v + w) == data .+ Int32(3)
            @test sum(v) == sum(Int64.(data))
        end
    end

    @testset "display" begin
        v = DpuVector(Int32.(collect(1:N)))
        buf = IOBuffer()
        show(buf, v)
        @test String(take!(buf)) == "DpuVector{Int32}($N)"

        show(buf, MIME("text/plain"), v)
        s = String(take!(buf))
        @test occursin("$N-element DpuVector{Int32}:", s)
        @test occursin("1", s)
    end

    # ---- RPN expression API ----

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
        UpmemVector.sync()

        before = UpmemVector.stat_compute_launches()
        fs = [reduce_expr(vs[i], vs[i + 1]) do x; sum(x[1] * x[2]) end
              for i in 1:5]
        vals = [get(f) for f in fs]
        passes = UpmemVector.stat_compute_launches() - before

        want = [sum(Int64.(collect(1:n) .+ i) .* Int64.(collect(1:n) .+ i .+ 1))
                for i in 1:5]
        @test vals == want
        # Five independent reductions left unread share one kernel pass.
        @test passes == 1
    end

    @testset "argmin / argmax over vectors" begin
        av = Int32[3, 1, 4, 1, 5, 9, 2, 6]
        bv = Int32[2, 7, 1, 8, 2, 8, 1, 8]
        cv = fill(Int32(5), 8)
        a = DpuVector(av); b = DpuVector(bv); c = DpuVector(cv)

        # 0-based winning lane, as the kernel produces it
        @test Array(argmin_of([a, b, c])) ==
              Int32[argmin([av[i], bv[i], cv[i]]) - 1 for i in 1:8]
        @test Array(argmax_of([a, b, c])) ==
              Int32[argmax([av[i], bv[i], cv[i]]) - 1 for i in 1:8]

        @test_throws ArgumentError argmin_of(DpuVector[])

        # the same thing spelled inside an expression
        @test Array(transform(a, b) do x
            argmin_lanes([x[1], x[2]])
        end) == Int32[av[i] <= bv[i] ? 0 : 1 for i in 1:8]
    end

    @testset "min_squared_distance" begin
        c1 = Int32[1, 5, 9, 2]; c2 = Int32[2, 6, 1, 7]
        q = Int32[4, 4]
        cols = [DpuVector(c1), DpuVector(c2)]
        want = minimum([(c1[i] - q[1])^2 + (c2[i] - q[2])^2 for i in 1:4])
        @test get(min_squared_distance(cols, q)) == want

        @test_throws ArgumentError min_squared_distance(cols, Int32[1])
        @test_throws ArgumentError min_squared_distance(DpuVector[], Int32[])
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

    # ---- lazy broadcasting ----

    @testset "broadcast lowers to one program" begin
        n = 512
        av = Int32.(collect(1:n)); bv = Int32.(collect(n:-1:1))
        cv = fill(Int32(3), n)
        a = DpuVector(av); b = DpuVector(bv); c = DpuVector(cv)

        @test Array(a .+ b .* c) == av .+ bv .* cv
        @test Array(abs.(a .- b) .+ 1) == abs.(av .- bv) .+ 1
        @test Array((a .+ 1) .* (b .- 2)) == (av .+ 1) .* (bv .- 2)
        @test Array(a .* a) == av .* av
        @test Array(.-a) == .-av
        @test Array(a .>> 2) == av .>> 2
        @test Array(2 .* a .+ 1) == 2 .* av .+ 1

        # comparisons inside a broadcast
        @test Array(a .> b) == Int32.(av .> bv)
        @test Array(a .<= b) == Int32.(av .<= bv)
        @test Array(a .== b) == Int32.(av .== bv)

        # ifelse is the broadcast spelling of select
        @test Array(ifelse.(a .> b, a, b)) == max.(av, bv)
    end

    @testset "a whole expression is one kernel pass" begin
        n = 512
        vs = [DpuVector(Int32.(collect(1:n) .+ k)) for k in 1:8]
        want = sum(Int32.(collect(1:n) .+ k) for k in 1:8)

        UpmemVector.sync()
        before = UpmemVector.stat_compute_launches()
        fused = UpmemVector.stat_vertical_fusions()
        got = Array(vs[1] .+ vs[2] .+ vs[3] .+ vs[4] .+
                    vs[5] .+ vs[6] .+ vs[7] .+ vs[8])
        passes = UpmemVector.stat_compute_launches() - before

        @test got == want
        # One program, so one pass -- and no reliance on the fusion pass to get
        # there.  The eager operator spelling of the same thing costs 7.
        @test passes == 1
        @test UpmemVector.stat_vertical_fusions() - fused == 0
    end

    @testset "in-place broadcast writes through" begin
        n = 512
        av = Int32.(collect(1:n)); bv = Int32.(collect(n:-1:1))
        a = DpuVector(av); b = DpuVector(bv)

        d = DpuVector(n)
        UpmemVector.sync()
        before = UpmemVector.stat_compute_launches()
        d .= a .+ b .* 2
        @test Array(d) == av .+ bv .* 2
        @test UpmemVector.stat_compute_launches() - before == 1

        # DpuVector is a handle type, so `.=` must update the buffer rather than
        # rebind -- an alias has to observe the write.
        alias = d
        d .= a .* 3
        @test Array(alias) == av .* 3

        # the destination may appear in its own expression
        c = DpuVector(copy(av))
        c .= c .+ 100
        @test Array(c) == av .+ 100

        e = DpuVector(copy(av))
        e .= e .* e
        @test Array(e) == av .* av

        @test_throws DimensionMismatch (DpuVector(8) .= a .+ b)
    end

    @testset "broadcast rejects what it cannot lower" begin
        a = DpuVector(Int32.(collect(1:64)))
        @test_throws ArgumentError Array(sqrt.(a))
        @test_throws ArgumentError Array(sin.(a))
    end

    @testset "similar and empty-runtime sync" begin
        a = DpuVector(Int32.(collect(1:N)))
        @test length(similar(a)) == N
        @test length(DpuVector(16)) == 16
        @test_throws ArgumentError DpuVector(-1)
        # sync() must be safe even with nothing submitted
        @test UpmemVector.sync() === nothing
    end

end
