const BEGIN_MARKER = "PARAM DEFAULTS BEGIN"
const END_MARKER = "PARAM DEFAULTS END"

const FIXED_ALIASES = Dict(
    "dim" => ("DIM", "dim"),
    "k" => ("K", "k"),
    "features" => ("FEATURES",),
    "classes" => ("CLASSES",),
    "depth" => ("DEPTH",),
    "bins" => ("BINS", "bins"),
    "scaling_shift" => ("scaling_shift", "prevent_overflow_shift_amount"),
    "learning_shift" => ("learning_shift",),
    "learning_rate" => ("learning_rate",),
)

function defaults_only(text::String, path::AbstractString)
    occursin(BEGIN_MARKER, text) || error("$path has no $BEGIN_MARKER marker")
    occursin(END_MARKER, text) || error("$path has no $END_MARKER marker")
    tail = split(text, BEGIN_MARKER; limit = 2)[2]
    body = split(tail, END_MARKER; limit = 2)[1]
    lines = split(strip(body, '\n'), '\n')
    !isempty(lines) && strip(lines[end]) in ("#", "//") && pop!(lines)
    return join(lines, '\n') * "\n"
end

function replace_first(text::String, pattern::Regex, replacement::AbstractString)
    found = Ref(false)
    updated = replace(text, pattern => matched -> begin
        found[] = true
        capture = match(pattern, matched).captures[1]
        string(something(capture, ""), replacement)
    end; count = 1)
    return updated, found[]
end

function set_parameter(text::String, name::AbstractString, value; julia::Bool)
    occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", name) ||
        error("invalid parameter $name")
    rendered = string(value)
    if julia
        pattern = Regex("(\\bconst\\s+$name\\s*=\\s*)([^#;\\n]+)")
        return replace_first(text, pattern, rendered)
    end

    assignment = Regex("(\\b$name\\s*=\\s*)([^;]+);")
    updated, found = replace_first(text, assignment, rendered * ";")
    found && return updated, true
    pattern = Regex("(?m)^(\\s*#define\\s+$name\\s+)([^\\n]*)")
    return replace_first(text, pattern, rendered)
end

function set_operation(text::String, operation::AbstractString; julia::Bool)
    if julia
        return replace_first(text, r"(operation\s*\(a,\s*b\)\s*=\s*)([^\n]+)",
                             "@. " * operation)
    end
    return replace_first(
        text, r"(?m)^(\s*#define\s+OPERATION\s*\([^\n]*?\)\s*)([^\n]*)",
        operation)
end

function set_aliases(text::String, names, value; julia::Bool)
    changed = false
    for name in names
        text, found = set_parameter(text, name, value; julia)
        changed |= found
    end
    return text, changed
end

function generated_parameters(source::AbstractString; elements::Int, dpus::Int,
                              warmup::Int, iterations::Int, check::Bool,
                              load_ref::Bool, seed::Int, fixed,
                              operation = nothing)
    julia = endswith(source, ".jl")
    text = defaults_only(read(source, String), source)
    values = (
        (("N", "nr_elements", "num_elements"), elements),
        (("dpu_number",), dpus),
        (("iterations", "iter"), iterations),
        (("warmup_iterations",), warmup),
        (("check_correctness",), Int(check)),
        # Independent of check_correctness: the sweep loads inputs from the
        # reference files so every variant's load stage is the same read,
        # whether or not results are verified afterwards.
        (("load_ref",), Int(load_ref || check)),
        (("seed",), seed),
    )
    for (names, value) in values
        text, _ = set_aliases(text, names, value; julia)
    end
    for (name, value) in fixed
        aliases = get(FIXED_ALIASES, lowercase(name), (name,))
        text, _ = set_aliases(text, aliases, value; julia)
    end
    if operation !== nothing
        text, found = set_operation(text, string(operation); julia)
        found || @warn "operation is absent from template" source
    end
    return text
end

function generated_path(source::AbstractString)
    base, extension = splitext(source)
    return base * ".generated" * extension
end
