# One operator at a time: vector-vector, vector-scalar, unary, and the
# comparisons that route through RPN.

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
