# APIs over K whole vectors at once.

@testset "per-element winner across vectors" begin
    av = Int32[3, 1, 4, 1, 5, 9, 2, 6]
    bv = Int32[2, 7, 1, 8, 2, 8, 1, 8]
    cv = fill(Int32(5), 8)
    a = DpuVector(av); b = DpuVector(bv); c = DpuVector(cv)

    # The same call Julia would make on the host arrays, element for element.
    @test Array(argmin.(zip(a, b, c))) == argmin.(zip(av, bv, cv))
    @test Array(argmax.(zip(a, b, c))) == argmax.(zip(av, bv, cv))
    @test Array(argmin.(zip(a, b))) == argmin.(zip(av, bv))
    @test Array(argmax.(zip(a))) == argmax.(zip(av))

    # Ties: Julia keeps the first, and cv ties with av at position 5.
    tv = fill(Int32(5), 8)
    t = DpuVector(tv)
    @test Array(argmax.(zip(t, t, t))) == argmax.(zip(tv, tv, tv))

    # findmin./findmax. yield a tuple per element; the device returns the two
    # columns unzipped, so zip them back to compare.
    values, labels = findmin_lanes([a, b, c])
    @test collect(zip(Array(values), Array(labels))) == findmin.(zip(av, bv, cv))
    values, labels = findmax_lanes([a, b, c])
    @test collect(zip(Array(values), Array(labels))) == findmax.(zip(av, bv, cv))

    # One pass for the pair when the build has room for both chains.
    sync(); before = PolymerPIM.stat_compute_launches()
    findmax_lanes([a, b, c]); sync()
    @test PolymerPIM.stat_compute_launches() - before == (MAX_CHAINS >= 2 ? 1 : 2)
    sync(); before = PolymerPIM.stat_compute_launches()
    argmax.(zip(a, b, c)); sync()
    @test PolymerPIM.stat_compute_launches() - before == 1

    @test_throws ArgumentError findmin_lanes(DpuVector[])
    # Any other broadcast over a zip would collect the vectors element by element.
    @test_throws ArgumentError sum.(zip(a, b))

    # The vertical form is unaffected: the index of the largest element.
    @test argmax(a) == argmax(av)

    # argmin_lanes inside an expression is the raw opcode, still 0-based.
    @test Array(transform(a, b) do x
        argmin_lanes([x[1], x[2]])
    end) == argmin.(zip(av, bv)) .- 1
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
