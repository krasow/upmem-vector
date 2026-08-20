# APIs over K whole vectors at once.

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
