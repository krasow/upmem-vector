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

        @test_throws ArgumentError apply!(x, 1, Ops.SCALAR_EQ)
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

end
