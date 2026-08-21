# PARAM DEFAULTS BEGIN
module Param
# Elementwise Benchmark Parameters
const T = Int32
const N = 805306368
const iterations = 20
const warmup_iterations = 5
const check_correctness = 1
const load_ref = 1
const ref_path = "../../cpu-verification/elementwise/data"
const seed = 1
operation(a, b) = @. -abs(((a) + (b)) - (a))
end
# PARAM DEFAULTS END
