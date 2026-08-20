# Writes that go through the existing DPU buffer instead of allocating.

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
    @test_throws Exception apply!(x, 1, PolymerPIM.Opcodes.OP_EQ_SCALAR)
end
