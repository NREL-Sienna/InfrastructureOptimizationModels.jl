bilinear_methods = (
    ("Bin2+sSOS", IOM._add_bin2_sos2_bilinear_approx!, ()),
    ("Bin2+mSOS+McQuad", IOM._add_bin2_manual_sos2_bilinear_approx!, (add_mccormick = true,)),
    ("Bin2+Saw", IOM._add_bin2_sawtooth_bilinear_approx!, ()),
    ("Bin2+DNMDT", IOM._add_bin2_dnmdt_bilinear_approx!, (double = true,)),
    ("Bin2+T-DNMDT", IOM._add_bin2_dnmdt_bilinear_approx!, (double = true, tighten = true,)),
    ("Bin2+DNMDT+McQuad", IOM._add_dnmdt_bilinear_approx!, (double = true, add_mccormick = true)),
    ("HybS+sSOS", IOM._add_hybs_sos2_bilinear_approx!, ()),
    ("HybS+sSOS+McAll", IOM._add_hybs_sos2_bilinear_approx!, (add_mccormick = true, add_quad_mccormick = true)),
    ("HybS+Saw", IOM._add_hybs_sawtooth_bilinear_approx!, ()),
    ("HybS+Saw+McAll", IOM._add_hybs_sawtooth_bilinear_approx!, (add_mccormick = true, add_quad_mccormick = true)),
    ("HybS+T-Saw", IOM._add_hybs_sawtooth_bilinear_approx!, (tighten = true,)),
    ("HybS+T-Saw+McBil", IOM._add_hybs_sawtooth_bilinear_approx!, (tighten = true, add_mccormick = true)),
    ("NMDT", IOM._add_dnmdt_bilinear_approx!, (double = false,)),
    ("DNMDT", IOM._add_dnmdt_bilinear_approx!, (double = true,)),
    ("DNMDT+McBil", IOM._add_dnmdt_bilinear_approx!, (double = true, add_mccormick = true))
)

if abspath(PROGRAM_FILE) == @__FILE__
    N = get(ARGS, 1, "10") |> x -> parse(Int, x)
    K = get(ARGS, 2, "3") |> x -> parse(Int, x)
    seed = get(ARGS, 3, "42") |> x -> parse(Int, x)
    run_benchmark(; N, K, seed)
end
