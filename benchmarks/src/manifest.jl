struct ManifestHandle
    path::String
    document::Dict{String,Any}
    index::Int
end

function write_toml(io::IO, document)
    buffer = IOBuffer()
    TOML.print(buffer, document; sorted = true)
    seekstart(buffer)
    previous_blank = true
    for raw in eachline(buffer)
        line = lstrip(raw)
        if startswith(line, '[') && !previous_blank
            println(io)
        end
        println(io, line)
        previous_blank = isempty(line)
    end
end

function save_manifest(handle::ManifestHandle)
    mkpath(dirname(handle.path))
    temporary = handle.path * ".tmp"
    open(temporary, "w") do io
        write_toml(io, handle.document)
    end
    mv(temporary, handle.path; force = true)
end

function archive_manifest(path::AbstractString)
    archived = path * ".legacy"
    index = 1
    while ispath(archived)
        index += 1
        archived = path * ".legacy.$index"
    end
    mv(path, archived)
    println("Archived legacy manifest: $archived")
end

function begin_manifest(path::AbstractString, kind::AbstractString, entry)
    document = if isfile(path)
        existing = TOML.parsefile(path)
        if get(existing, "version", 0) == 1
            archive_manifest(path)
            Dict{String,Any}("version" => 2, "kind" => kind,
                             "invocations" => Dict{String,Any}[])
        else
            existing
        end
    else
        Dict{String,Any}("version" => 2, "kind" => kind,
                         "invocations" => Dict{String,Any}[])
    end
    get(document, "version", 0) == 2 || error("unsupported manifest $path")
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

function manifest_dimensions(cases::Vector{RunCase}, variants; profile = nothing)
    isempty(cases) && error("manifest dimensions require at least one case")
    case = first(cases)
    all(item -> item.benchmark == case.benchmark &&
                item.warmup == case.warmup &&
                item.iterations == case.iterations &&
                item.check == case.check &&
                item.seed == case.seed &&
                item.parameters == case.parameters &&
                item.operation == case.operation, cases) ||
        error("manifest cases do not form one benchmark sweep")
    entry = Dict{String,Any}(
        "benchmark" => case.benchmark,
        "dpus" => unique(item.dpus for item in cases),
        "elements_per_dpu" => unique(item.elements_per_dpu for item in cases),
        "warmup" => case.warmup,
        "iterations" => case.iterations,
        "seed" => case.seed,
        "variants" => variants,
    )
    isempty(case.parameters) || (entry["parameters"] = case.parameters)
    case.operation === nothing || (entry["operation"] = case.operation)
    if profile !== nothing
        entry["fusion_profile"] = profile.path
        entry["fusion_build"] = profile.build
    end
    return entry
end
