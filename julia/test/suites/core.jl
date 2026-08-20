# Construction, host transfers, indexing, display, and the plumbing
# Julia expects from a container type.

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

@testset "similar and empty-runtime sync" begin
    a = DpuVector(Int32.(collect(1:N)))
    @test length(similar(a)) == N
    @test length(DpuVector(16)) == 16
    @test_throws ArgumentError DpuVector(-1)
    # sync() must be safe even with nothing submitted
    @test UpmemVector.sync() === nothing
end
