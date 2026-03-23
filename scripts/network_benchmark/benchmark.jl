"""
Benchmark script for the bilinear delta model described in `bilinear_delta_model.tex`.

Builds a network OPF with delta (incremental) PWL costs and bilinear V·I power
balance constraints. Compares three Bin2 bilinear approximation methods from
InfrastructureOptimizationModels:

  1. Solver-native SOS2
  2. Manual SOS2 (binary-variable adjacency)
  3. Sawtooth MIP

Usage:
    julia --project=test scripts/bilinear_delta_benchmark.jl [N] [K] [seed]

    N    = number of nodes (default 10, must be even)
    K    = number of PWL cost segments per generator (default 3)
    seed = random seed for network generation (default 42)
"""

using InfrastructureOptimizationModels
using JuMP
using HiGHS
using Dates
using Printf

const IOM = InfrastructureOptimizationModels
const IS = IOM.IS

# Try to load Ipopt for NLP reference; skip if unavailable
const HAS_IPOPT = try
    @eval using Ipopt
    true
catch
    false
end

include("problem.jl")
include("formulations.jl")
include("nlp.jl")

# ─── Metrics ──────────────────────────────────────────────────────────────────

function compute_bilinear_residuals(result, net::LosslessNetworkProblem)
    V = result.V_container
    I = result.I_container
    residuals = Float64[]
    grouped_nodes = (net.gen_nodes, net.dem_nodes)
    zs = (result.z_gen, result.z_dem)
    for (nodes, z) in zip(grouped_nodes, zs)
        for n in nodes
            product = JuMP.value(V[n, 1]) * JuMP.value(I[n, 1])
            if product == 0.0
                continue
            end
            resid = abs(product - JuMP.value(z[n, 1])) / abs(product)
            push!(residuals, resid)
        end
    end
    geometric_mean = reduce(*, residuals)^(1 / length(residuals))
    return geometric_mean, maximum(residuals)
end

function compute_bilinear_residuals(result, net::LossyNetworkProblem)
    V, I = result.V_container, result.I_container
    residuals = Float64[]
    for node in net.gen_nodes
        V_val = JuMP.value(V[node, 1])
        I_val = JuMP.value(I[node, 1])
        a, b, c = net.loss_coeffs[node]
        gt = V_val * I_val - a*I_val^2 - b*I_val - c
        resid = abs(gt - JuMP.value(result.VI_gen[node, 1])) / abs(gt)
        push!(residuals, resid)
    end
    for node in net.dem_nodes
        gt = JuMP.value(V[node, 1]) * JuMP.value(I[node, 1])
        resid = abs(gt - JuMP.value(result.VI_dem[node, 1])) / abs(gt)
        push!(residuals, resid)
    end
    geometric_mean = reduce(*, residuals)^(1 / length(residuals))
    return geometric_mean, maximum(residuals)
end

function model_size(jump_model)
    nv = JuMP.num_variables(jump_model)
    nc = sum(
        JuMP.num_constraints(jump_model, f, s)
        for (f, s) in JuMP.list_of_constraint_types(jump_model)
    )
    nb = JuMP.num_constraints(jump_model, JuMP.VariableRef, JuMP.MOI.ZeroOne)
    return (; variables = nv, constraints = nc, binaries = nb)
end

# ─── Main benchmark ──────────────────────────────────────────────────────────

"""
    run_benchmark(; N, K, seed, segment_counts, sawtooth_depths)

Run the full benchmark comparing the three Bin2 bilinear approximation methods.

# Keyword arguments
- `N::Int = 10`: number of nodes (must be even)
- `K::Int = 3`: number of PWL cost segments per generator
- `seed::Int = 42`: random seed for network generation
- `segment_counts::Vector{Int} = [2, 4, 8, 16]`: segment counts for SOS2 methods
- `sawtooth_depths::Vector{Int} = [2, 4, 8]`: depths for sawtooth method
"""
function run_benchmark(
    bilinear_methods,
    refinements;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
    # segment_counts::Vector{Int} = [2, 4, 8],
    # sawtooth_depths::Vector{Int} = [2, 4, 8],
)
    net = generate_lossy_network(; N, K, seed)
    println("Network: $(net.N) nodes, $(length(net.edges)) edges, $(net.K) cost segments")
    println("Generators: $(length(net.gen_nodes)), Demands: $(length(net.dem_nodes))")
    println()

    # --- NLP reference ---
    nlp_obj = NaN
    if false#HAS_IPOPT
        println("="^100)
        println("NLP Reference (Ipopt, bilinear V·I constraints)")
        println("="^100)
        nlp_model = build_vi_formulation(net, _add_bilinear_nlp!, Dict(), 0).jump_model
        # nlp_model = build_vi_formulation(net, _)
        JuMP.set_optimizer(nlp_model, Ipopt.Optimizer)
        JuMP.set_optimizer_attribute(nlp_model, "print_level", 0)
        nlp_t = @elapsed JuMP.optimize!(nlp_model)
        nlp_status = JuMP.termination_status(nlp_model)
        sz = model_size(nlp_model)
        if nlp_status == JuMP.LOCALLY_SOLVED
            nlp_obj = JuMP.objective_value(nlp_model)
        end
        @printf("  Status:      %s\n", nlp_status)
        @printf("  Objective:   %.6f\n", nlp_obj)
        @printf("  Variables:   %d\n", sz.variables)
        @printf("  Constraints: %d\n", sz.constraints)
        @printf("  Solve time:  %.4f s\n", nlp_t)
    else
        println("Ipopt not available — skipping NLP reference.")
        println("Install Ipopt.jl in the test environment for NLP comparison.")
    end
    println()

    # --- MIP benchmarks ---
    println("="^100)
    println("MIP Bilinear Approximations (HiGHS)")
    println("  Refinement = num_segments for SOS2 methods, depth for Sawtooth")
    println("="^100)
    @printf("%-17s %4s %6s %7s %6s %12s %9s %11s %10s %8s\n",
        "Method", "Ref", "Vars", "Constrs", "Bins", "Objective", "Gap(%)", "Mean Resid",
        "Max Resid", "Time(s)")
    println("-"^100)

    for (label, fn, kw) in bilinear_methods
        for ref in refinements
            build_t = @elapsed begin
                result = build_vi_formulation(net, fn, kw, ref)
            end

            JuMP.set_optimizer(result.jump_model, HiGHS.Optimizer)
            JuMP.set_optimizer_attribute(result.jump_model, "log_to_console", false)
            JuMP.set_optimizer_attribute(result.jump_model, "time_limit", 300.0)

            solve_t = @elapsed JuMP.optimize!(result.jump_model)
            status = JuMP.termination_status(result.jump_model)
            sz = model_size(result.jump_model)

            if status in (JuMP.OPTIMAL, JuMP.TIME_LIMIT) &&
               JuMP.has_values(result.jump_model)
                obj = JuMP.objective_value(result.jump_model)
                gap = isnan(nlp_obj) ? NaN : abs(nlp_obj - obj) / abs(nlp_obj) * 100.0
                geometric_mean, max = compute_bilinear_residuals(result, net)
                gap_str = isnan(gap) ? "    -" : @sprintf("%8.4f", gap)
                @printf("%-17s %4d %6d %7d %6d %12.6f %8s %11.2e %10.2e %8.4f\n",
                    label, ref,
                    sz.variables, sz.constraints, sz.binaries,
                    obj, gap_str, geometric_mean, max, solve_t)
            else
                @printf("%-15s %4d %6d %7d %6d %12s %9s %10s %8.4f\n",
                    label, ref,
                    sz.variables, sz.constraints, sz.binaries,
                    string(status), "-", "-", solve_t)
            end
        end
        println()
    end
    println("="^100)
end

# ─── Entry point ──────────────────────────────────────────────────────────────
