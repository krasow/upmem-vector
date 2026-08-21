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

    # findmin. yields a tuple per element; the device unzips, so zip it back.
    values, labels = findmin_lanes([a, b, c])
    @test collect(zip(Array(values), Array(labels))) == findmin.(zip(av, bv, cv))
    values, labels = findmax_lanes([a, b, c])
    @test collect(zip(Array(values), Array(labels))) == findmax.(zip(av, bv, cv))

    # Lazy like any other broadcast: one program, not a launch then a scale.
    sync(); before = PolymerPIM.stat_compute_launches()
    scaled = Array(argmin.(zip(a, b, c)) .* Int32(11) .+ Int32(2))
    sync()
    @test PolymerPIM.stat_compute_launches() - before == 1
    @test scaled == argmin.(zip(av, bv, cv)) .* 11 .+ 2

    # And the lanes themselves may be expressions.
    @test Array(argmax.(zip(a .+ b, c))) == argmax.(zip(av .+ bv, cv))

    # A reduction over one still folds into a single program.
    @test sum(argmin.(zip(a, b, c)))[] == sum(argmin.(zip(av, bv, cv)))

    # One pass for the pair when the build has room for both chains.
    sync(); before = PolymerPIM.stat_compute_launches()
    v, l = findmax_lanes([a, b, c]); Array(v); Array(l); sync()
    @test PolymerPIM.stat_compute_launches() - before == (MAX_CHAINS >= 2 ? 1 : 2)
    sync(); before = PolymerPIM.stat_compute_launches()
    Array(argmax.(zip(a, b, c))); sync()
    @test PolymerPIM.stat_compute_launches() - before == 1

    @test_throws ArgumentError findmin_lanes(DpuVector[])
    # Any other broadcast over a zip would collect the vectors element by element.
    @test_throws ArgumentError sum.(zip(a, b))

    # The vertical form is unaffected: the index of the largest element.
    @test argmax(a) == argmax(av)

    # The same call inside a hand-built program, over the expressions themselves.
    @test Array(transform(a, b) do x
        argmin([x[1], x[2]])
    end) == argmin.(zip(av, bv))
    @test Array(transform(a, b) do x
        argmax(zip(x[1], x[2]))
    end) == argmax.(zip(av, bv))
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
