include("benchmark.jl")

bilinear_methods = (
    ("Bin2+sSOS", IOM._add_bin2_sos2_bilinear_approx!, ()),
    ("Bin2+Saw", IOM._add_bin2_sawtooth_bilinear_approx!, ()),
    ("HybS+sSOS", IOM._add_hybs_sos2_bilinear_approx!, ()),
    ("HybS+Saw", IOM._add_hybs_sawtooth_bilinear_approx!, ()),
)

refinements = [2]

if abspath(PROGRAM_FILE) == @__FILE__
    N = get(ARGS, 1, "10") |> x -> parse(Int, x)
    K = get(ARGS, 2, "3") |> x -> parse(Int, x)
    seed = get(ARGS, 3, "42") |> x -> parse(Int, x)
    run_benchmark(bilinear_methods, refinements; N, K, seed)
end
