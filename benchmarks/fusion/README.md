# Fusion profiles

`tune.jl` writes `<benchmark>.toml` files here. Each profile records the best
fusion build parameters and the cases used to measure them. `runner.jl`
automatically applies a matching profile when PolymerPIM or Julia is selected.

Profiles contain these build-time knobs:

- `FUSION_LOOKAHEAD`
- `MAX_HFUSE_CHAINS`
- `JIT_BATCH_SIZE`
- `MAX_VFUSE_OPS`
- `MAX_VFUSE_INPUTS`
- `BLOCK_SIZE_LOG2`

Use `runner.jl --no-profile` to run with the Makefile defaults.
