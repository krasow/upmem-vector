# PARAM DEFAULTS BEGIN
module Param
# Histogram Benchmark Parameters
const T = Int32
const DEPTH = 12   # input values span 2^DEPTH
const BINS = 256
const N = 2147483648
const iterations = 1
const warmup_iterations = 1
const check_correctness = 0
const load_ref = 1
const ref_path = "../../cpu-verification/hist/data"
const seed = 1
end
# PARAM DEFAULTS END
