struct ManifestHandle
    path::String
    document::Dict{String,Any}
    index::Int
end

function save_manifest(handle::ManifestHandle)
    mkpath(dirname(handle.path))
    temporary = handle.path * ".tmp"
    open(temporary, "w") do io
        TOML.print(io, handle.document; sorted = true)
    end
    mv(temporary, handle.path; force = true)
end

function begin_manifest(path::AbstractString, kind::AbstractString, entry)
    document = isfile(path) ? TOML.parsefile(path) :
               Dict{String,Any}("version" => 1, "kind" => kind,
                                "invocations" => Dict{String,Any}[])
    get(document, "version", 0) == 1 || error("unsupported manifest $path")
    get(document, "kind", "") == kind || error("manifest kind mismatch in $path")
    invocation = Dict{String,Any}(entry)
    invocation["started_at"] = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ")
    invocation["status"] = "running"
    push!(document["invocations"], invocation)
    handle = ManifestHandle(string(path), document, length(document["invocations"]))
    save_manifest(handle)
    return handle
end

function finish_manifest(handle::ManifestHandle, status::AbstractString;
                         failure = nothing)
    invocation = handle.document["invocations"][handle.index]
    invocation["status"] = status
    invocation["finished_at"] = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ")
    failure === nothing || (invocation["error"] = string(failure))
    save_manifest(handle)
end

function manifest_case(case::RunCase, variants; profile = nothing)
    entry = Dict{String,Any}(
        "benchmark" => case.benchmark,
        "dpus" => case.dpus,
        "elements_per_dpu" => case.elements_per_dpu,
        "total_elements" => total_elements(case),
        "warmup" => case.warmup,
        "iterations" => case.iterations,
        "check" => case.check,
        "seed" => case.seed,
        "variants" => variants,
        "parameters" => case.parameters,
    )
    case.operation === nothing || (entry["operation"] = case.operation)
    if profile !== nothing
        entry["fusion_profile"] = profile.path
        entry["fusion_build"] = profile.build
    end
    return entry
end
