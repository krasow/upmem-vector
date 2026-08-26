function test_case(; benchmark = "elementwise", dpus = 2,
                   elements_per_dpu = 64, warmup = 0, iterations = 1,
                   check = false, load_ref = true, seed = 1,
                   parameters = Dict{String,Any}(), operation = nothing)
    return BenchmarkRunner.RunCase(
        benchmark, dpus, elements_per_dpu, warmup, iterations, check,
        load_ref, seed, parameters, operation)
end

function csv_records(path)
    lines = readlines(path)
    isempty(lines) && return Dict{String,String}[]
    columns = split(first(lines), ','; keepempty = true)
    return [Dict(zip(columns, split(line, ','; keepempty = true)))
            for line in Iterators.drop(lines, 1)]
end

function captured_output(f)
    return mktemp() do _, output
        redirect_stdout(output) do
            redirect_stderr(output) do
                f()
            end
        end
        flush(output)
        seekstart(output)
        read(output, String)
    end
end

function quietly(f)
    result = Ref{Any}()
    captured_output() do
        result[] = f()
    end
    return result[]
end
