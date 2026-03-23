bilinear_methods = (
    (
        "Bin2+sSOS",
        SeparableMethod(),
        IOM._add_sos2_bilinear_approx!,
        (),
        IOM._add_sos2_quadratic_approx!,
    ),
    (
        "Bin2+Saw",
        SeparableMethod(),
        IOM._add_sawtooth_bilinear_approx!,
        (),
        IOM._add_sawtooth_bilinear_approx!,
    ),
    (
        "HybS+sSOS",
        SeparableMethod(),
        IOM._add_hybs_sos2_bilinear_approx!,
        (),
        IOM._add_sos2_quadratic_approx!,
    ),
    (
        "HybS+Saw",
        SeparableMethod(),
        IOM._add_hybs_sawtooth_bilinear_approx!,
        (),
        IOM._add_sawtooth_bilinear_approx!,
    ),
    (
        "DNMDT",
        DNMDTMethod(),
        IOM._add_dnmdt_bilinear_approx!,
        (),
        IOM._add_dnmdt_quadratic_approx!,
    ),
)

include("bilinear_delta_benchmark.jl")

# ─── Entry point ──────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    N = get(ARGS, 1, "10") |> x -> parse(Int, x)
    K = get(ARGS, 2, "3") |> x -> parse(Int, x)
    seed = get(ARGS, 3, "42") |> x -> parse(Int, x)
    run_benchmark(LosslessNetworkProblem; N, K, seed)
    if "--lossy" in ARGS
        println("\n")
        run_benchmark(LossyNetworkProblem; N, K, seed)
    end
end
