# Broadcasting.  The Broadcasted tree stays lazy and lowers to a single RPN
# program, so these also pin the one-pass-per-expression property.

@testset "single operator broadcasts" begin
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

    PolymerPIM.sync()
    before = PolymerPIM.stat_compute_launches()
    fused = PolymerPIM.stat_vertical_fusions()
    got = Array(vs[1] .+ vs[2] .+ vs[3] .+ vs[4] .+
                vs[5] .+ vs[6] .+ vs[7] .+ vs[8])
    passes = PolymerPIM.stat_compute_launches() - before

    @test got == want
    # One program, so one pass -- and no reliance on the fusion pass to get
    # there.  The eager operator spelling of the same thing costs 7.
    @test passes == 1
    @test PolymerPIM.stat_vertical_fusions() - fused == 0
end

@testset "in-place broadcast writes through" begin
    n = 512
    av = Int32.(collect(1:n)); bv = Int32.(collect(n:-1:1))
    a = DpuVector(av); b = DpuVector(bv)

    d = DpuVector(n)
    PolymerPIM.sync()
    before = PolymerPIM.stat_compute_launches()
    d .= a .+ b .* 2
    @test Array(d) == av .+ bv .* 2
    @test PolymerPIM.stat_compute_launches() - before == 1

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
    # Lazily built, so it raises at first use -- and only once: a failed
    # expression must not be retried by a later, unrelated sync().
    bad = sqrt.(a)
    @test_throws ArgumentError Array(bad)
    @test sync() === nothing
    @test_throws ArgumentError Array(sin.(a))
end

# abs2 lowers to `sqr`, which is OP_DUP + OP_MUL, so its argument is computed
# once.  Spelled `x .* x` the lowering has no CSE and computes it twice -- which
# is why knn uses abs2.
@testset "abs2 loads its argument once" begin
    a = DpuVector(Int32.(1:N)); b = DpuVector(Int32.(N:-1:1))
    O = PolymerPIM.Opcodes

    sq = @code_jitted abs2.(a .- b)
    @test O.OP_DUP in sq.ops
    @test count(==(O.OP_SUB), sq.ops) == 1

    naive = @code_jitted (a .- b) .* (a .- b)
    @test count(==(O.OP_SUB), naive.ops) == 2
    @test !(O.OP_DUP in naive.ops)
    @test length(sq.ops) < length(naive.ops)

    # The same program as the hand-built RPN it replaces.
    xs = DpuExpr[input(), operand(1)]
    byhand = code_jitted(sqr(xs[1] - xs[2]); nelements = length(a), noperands = 1)
    @test sq.ops == byhand.ops
    @test sq.hash == byhand.hash

    # No abs2 opcode exists: it is those two.
    @test !isdefined(O, :OP_ABS2)

    @test Array(abs2.(a .- b)) == (Array(a) .- Array(b)) .^ 2
end
