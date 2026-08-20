# REPL display for DpuVector

const _MAX_DISPLAY_ELEMENTS = 10

function Base.show(io::IO, v::DpuVector)
    print(io, "DpuVector{Int32}(", v.len, ")")
end

function Base.show(io::IO, ::MIME"text/plain", v::DpuVector)
    println(io, v.len, "-element DpuVector{Int32}:")
    if v.len == 0
        return
    end

    data = Array(v)

    if v.len <= 2 * _MAX_DISPLAY_ELEMENTS
        # small enough to show everything
        for i in eachindex(data)
            print(io, " ", data[i])
            i < length(data) && println(io)
        end
    else
        # show first and last _MAX_DISPLAY_ELEMENTS entries
        for i in 1:_MAX_DISPLAY_ELEMENTS
            println(io, " ", data[i])
        end
        println(io, " \u22ee")                  # vertical ellipsis
        for i in (v.len - _MAX_DISPLAY_ELEMENTS + 1):v.len
            print(io, " ", data[i])
            i < v.len && println(io)
        end
    end
end

_field(io, key, value) = println(io, "  ", rpad(key * ":", 16), value)

"""
    PolymerPIM.versioninfo(io = stdout)

Print the runtime shape, the configuration of the `libvectordpu` in use, and
where it came from. `configuration()` and `installinfo()` return the same
information as dictionaries.
"""
function versioninfo(io::IO = stdout)
    config, install = configuration(), installinfo()
    cfg(key, fallback = "unknown") = get(config, key, fallback)
    inst(key, fallback = "unknown") = get(install, key, fallback)

    println(io, "PolymerPIM v", pkgversion(@__MODULE__))

    println(io, "Runtime:")
    _field(io, "DPUs", string(ndpus(), runtime_initialized() ? "" : " (not yet claimed)"))
    _field(io, "Tasklets/DPU", ntasklets())
    _field(io, "Block size", string(1 << parse(Int, cfg("BLOCK_SIZE_LOG2", "0")),
                                    " elements/tasklet"))

    println(io, "Library:")
    _field(io, "Backend", cfg("BACKEND"))
    _field(io, "Pipeline/JIT", string("PIPELINE=", cfg("PIPELINE"), " JIT=", cfg("JIT"),
                                      " batch=", cfg("JIT_BATCH_SIZE")))
    _field(io, "Build", string(inst("BUILD_TYPE"), ", ", cfg("CXX_STANDARD"),
                               ", ", inst("CXX")))
    _field(io, "Fusion", string("lookahead=", cfg("FUSION_LOOKAHEAD"),
                                " chains=", cfg("MAX_HFUSE_CHAINS"),
                                " ops=", cfg("MAX_VFUSE_OPS")))
    _field(io, "Slots", string(MAX_VFUSE_INPUTS, " operands, ",
                               MAX_PIPELINE_SCALARS, " scalars, ",
                               MAX_LOCAL_SCRATCH_VECTORS, " locals"))

    println(io, "Install:")
    _field(io, "Prefix", inst("VECTORDPU_DIR"))
    _field(io, "Installed", string(inst("INSTALL_DATE"), " from ",
                                   inst("GIT_BRANCH"), " ", first(inst("GIT_REV"), 7),
                                   inst("GIT_DIRTY") == "1" ? " (dirty)" : ""))
    _field(io, "Wrapper", string(INSTALL_DIR, ", built ", inst("WRAPPER_BUILT")))
    _field(io, "Julia", string(inst("JULIA_VERSION"),
                               inst("JULIA_VERSION") == string(VERSION) ? "" :
                               " (running $(VERSION))"))
    return nothing
end
