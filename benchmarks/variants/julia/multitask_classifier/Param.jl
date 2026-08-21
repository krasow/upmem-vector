# PARAM DEFAULTS BEGIN
module Param
# Multitask Classifier (one-vs-rest linear SVM) Benchmark Parameters
const T = Int32
const N = 16777216
const FEATURES = 8
const CLASSES = 4
const iterations = 10
const warmup_iterations = 1
const check_correctness = 0
const load_ref = 1
const ref_path = "../../cpu-verification/multitask_classifier/data"
const seed = 1
end
# PARAM DEFAULTS END
