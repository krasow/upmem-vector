"""
    PolymerPIM.Internal

Low-level RPN builders and launch helpers used to implement PolymerPIM's lazy
array API. These interfaces are available for development and testing but are
not part of the stable public API.
"""
module Internal

const Parent = parentmodule(@__MODULE__)
const MAX_VFUSE_INPUTS = Parent.MAX_VFUSE_INPUTS
const MAX_PIPELINE_SCALARS = Parent.MAX_PIPELINE_SCALARS

include("opcodes.jl")
include("expr.jl")
include("pipelines.jl")

end
