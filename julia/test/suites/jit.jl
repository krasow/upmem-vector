# @code_jitted: the generated kernel for an expression, without running it.

@testset "code_jitted describes the lowered program" begin
    a = DpuVector(Int32.(1:N))
    b = DpuVector(Int32.(N+1:2N))
    c = DpuVector(fill(Int32(3), N))

    code = @code_jitted a .+ b .* c
    @test code isa PolymerPIM.JittedCode
    @test length(code.ops) == 5          # one program, not one per operator
    @test code.noperands == 2
    @test code.nelements == N
    @test occursin("k_" * code.hash, code.source)
    @test endswith(code.path, "k_" * code.hash * ".c")

    # Distinct expressions get distinct cache keys; the same one is stable.
    @test (@code_jitted a .+ b).hash != code.hash
    @test (@code_jitted a .+ b).hash == (@code_jitted a .+ b).hash
end

@testset "code_jitted covers the broadcast forms" begin
    a = DpuVector(Int32.(1:N))
    b = DpuVector(Int32.(N+1:2N))
    d = DpuVector(zeros(Int32, N))

    for code in (@code_jitted(abs.(a .- b) .+ 1),
                 @code_jitted(ifelse.(a .> b, a, b)),
                 @code_jitted(d .= a .* 2 .+ b))
        @test !isempty(code.source)
        @test occursin("int k_", code.source)
    end

    # Statically compiled kernels have no generated source.
    @test_throws ErrorException @code_jitted a + b
    @test_throws ErrorException @code_jitted sum(a)
end

@testset "reductions over a broadcast are one program" begin
    a = DpuVector(Int32.(1:N))
    b = DpuVector(Int32.(N+1:2N))

    code = @code_jitted sum(a .+ b)
    @test length(code.ops) == 4          # push, push, add, reduce
    @test code.ops[end] == PolymerPIM.Internal.Opcodes.OP_SUM
    @test occursin("acc_0", code.source)  # a reduction chain, not a store

    # An assignment describes its right-hand side; without stripping it first the
    # macro missed the reduction, evaluated it, and got a materialised Int.
    @test (@code_jitted g = sum(a .+ b)).hash == code.hash
    @test !@isdefined(g)
    @test (@code_jitted g = a .+ b).hash == (@code_jitted a .+ b).hash

    # Written literally: the macro matches the reduction by name, as macros do.
    @test length((@code_jitted prod(a .+ b)).ops) == 4
    @test length((@code_jitted minimum(a .+ b)).ops) == 4
    @test length((@code_jitted maximum(a .+ b)).ops) == 4

    # The macro must not run anything, even for the eager-looking form.
    before = PolymerPIM.stat_compute_launches()
    @code_jitted sum(a .- b)
    @test PolymerPIM.stat_compute_launches() == before

    # The mapped forms build the same program from a traced function.
    @test (@code_jitted sum(abs, a)).ops == UInt8[30, 2, 25]
    @test (@code_jitted mapreduce(abs, +, a)).hash == (@code_jitted sum(abs, a)).hash
    @test (@code_jitted mapreduce(abs, max, a)).ops[end] ==
          PolymerPIM.Internal.Opcodes.OP_MAX

    # sum.(a .+ b) is elementwise sum, not a reduction, and is not lowerable.
    @test_throws ArgumentError @code_jitted sum.(a .+ b)

    # sum(a) is a static reduce kernel, not a generated one.
    @test_throws ErrorException @code_jitted sum(a)
end

@testset "kernels declare only the slots they use" begin
    a = DpuVector(Int32.(1:N))
    b = DpuVector(Int32.(N+1:2N))
    src = (@code_jitted a .+ b).source

    # One result, one operand: the placeholders for every other slot the
    # interpreter needs are the interpreter's business, not this kernel's.
    @test occursin("res_ptrs[1];", src)
    @test occursin("res_blks[1];", src)
    @test occursin("op_blks[1];", src)
    @test !occursin("extra_res_offsets", src)
    @test !occursin("MAX_PIPELINE_SCALARS", src)   # no 128-entry scalar table
    @test !occursin("for (int k", src)             # no loops over MAX_ bounds
end

@testset "code_jitted matches what the JIT compiles" begin
    a = DpuVector(Int32.(1:N))
    b = DpuVector(Int32.(N+1:2N))

    code = @code_jitted a .- b .+ 7
    @test !iscompiled(code)              # nothing on disk until it is launched
    @test occursin("not compiled yet", sprint(show, MIME"text/plain"(), code))

    Array(a .- b .+ 7)                   # launching it compiles that kernel

    @test iscompiled(code)
    @test read(code.path, String) == code.source
    @test occursin("(compiled)", sprint(show, MIME"text/plain"(), code))
end

# The arg forms launch as they build, so the macro rebuilds their program rather
# than describing the vector that came back.
@testset "code_jitted covers the arg forms" begin
    a = DpuVector(Int32.(1:N)); b = DpuVector(Int32.(N:-1:1))

    # The lane node lowers like any broadcast, so the macro has no special case.
    lanes = @code_jitted argmax.(zip(a, b))
    @test PolymerPIM.Internal.Opcodes.OP_ARGMAX_K in lanes.ops
    @test lanes.noperands == 1
    @test lanes.nelements == N
    @test PolymerPIM.Internal.Opcodes.OP_ARGMIN_K in (@code_jitted argmin.(zip(a, b))).ops
    @test length((@code_jitted argmax.(zip(a, b)) .* 3).ops) > length(lanes.ops)

    # The vertical form has a kernel too now: the index pass.
    idx = @code_jitted argmax(a)
    @test PolymerPIM.Internal.Opcodes.OP_PUSH_GLOBAL_INDEX in idx.ops
    @test idx.ops[end] == PolymerPIM.Internal.Opcodes.OP_MIN
    @test (@code_jitted findmin(a)).hash == idx.hash

end
