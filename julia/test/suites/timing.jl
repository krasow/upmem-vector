@testset "synchronized timing" begin
    a = DPUVector(Int32.(collect(1:N)))
    sync()

    before = PolymerPIM.stat_compute_launches()
    result, output = mktemp() do _, io
        value = redirect_stdout(io) do
            @dputime a .+ 7
        end
        flush(io)
        seekstart(io)
        return value, read(io, String)
    end
    after = PolymerPIM.stat_compute_launches()

    @test result isa DpuLazy
    @test occursin(r"^DPU time: [0-9.]+ ms\n$", output)
    @test after > before

    sync()
    @test PolymerPIM.stat_compute_launches() == after
    @test Array(result) == Int32.(collect(1:N)) .+ 7

    scope = Module(:DpuTimeScopeTest)
    Core.eval(scope, :(using PolymerPIM))
    redirect_stdout(devnull) do
        Core.eval(scope, :(@dputime assigned = 42))
    end
    @test Core.eval(scope, :assigned) == 42
end
