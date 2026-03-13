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
using Random
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

# ─── Minimal types for IOM container infrastructure ──────────────────────────

mutable struct BenchmarkSystem <: IS.InfrastructureSystemsContainer
    base_power::Float64
end

IOM.get_base_power(sys::BenchmarkSystem) = sys.base_power
IOM.stores_time_series_in_memory(::BenchmarkSystem) = false

struct NetworkNode <: IS.InfrastructureSystemsComponent end

struct VoltageVariable <: IOM.VariableType end
struct CurrentVariable <: IOM.VariableType end

# ─── Network data ────────────────────────────────────────────────────────────

struct NetworkData
    N::Int
    K::Int
    gen_nodes::Vector{String}
    dem_nodes::Vector{String}
    all_nodes::Vector{String}
    edges::Vector{Tuple{String, String}}
    conductances::Dict{Tuple{String, String}, Float64}
    demands::Dict{String, Float64}
    marginal_costs::Dict{String, Vector{Float64}}
    segment_widths::Dict{String, Vector{Float64}}
end

"""
Generate a random connected network matching the model in bilinear_delta_model.tex.

- `N` nodes split evenly into generators and demands.
- `K` PWL cost segments per generator with nondecreasing marginal costs.
- Connected graph via random spanning tree plus extra edges.
- Conductances scaled for feasibility given voltage/current bounds.
- Demands d_i ∈ [0.05, 0.15].
"""
function generate_network(; N::Int = 10, K::Int = 3, seed::Int = 42)
    @assert iseven(N) "N must be even"
    rng = MersenneTwister(seed)

    all_nodes = ["n$(i)" for i in 1:N]
    gen_nodes = all_nodes[1:(N ÷ 2)]
    dem_nodes = all_nodes[(N ÷ 2 + 1):N]

    edges = Tuple{String, String}[]
    conductances = Dict{Tuple{String, String}, Float64}()

    # Random spanning tree: shuffle nodes, connect consecutive pairs
    perm = shuffle(rng, 1:N)
    for idx in 1:(N - 1)
        a, b = all_nodes[perm[idx]], all_nodes[perm[idx + 1]]
        e = a < b ? (a, b) : (b, a)
        if e ∉ edges
            push!(edges, e)
            conductances[e] = 1.0 + 4.0 * rand(rng)
        end
    end

    # Extra edges for density
    for _ in 1:(N ÷ 2)
        i, j = rand(rng, 1:N), rand(rng, 1:N)
        i == j && continue
        a, b = all_nodes[i], all_nodes[j]
        e = a < b ? (a, b) : (b, a)
        if e ∉ edges
            push!(edges, e)
            conductances[e] = 1.0 + 4.0 * rand(rng)
        end
    end

    # Demands at demand nodes (small enough for network to transport)
    demands = Dict(d => 0.05 + 0.1 * rand(rng) for d in dem_nodes)

    # PWL cost data: K segments per generator with nondecreasing marginal costs
    marginal_costs = Dict{String, Vector{Float64}}()
    segment_widths = Dict{String, Vector{Float64}}()
    for g in gen_nodes
        mc = sort(rand(rng, K) .* 10.0 .+ 1.0)
        marginal_costs[g] = mc
        widths = rand(rng, K) .+ 0.1
        widths .*= 1.5 / sum(widths)
        segment_widths[g] = widths
    end

    return NetworkData(
        N, K, gen_nodes, dem_nodes, all_nodes,
        edges, conductances, demands,
        marginal_costs, segment_widths,
    )
end

"""Build adjacency list from edge set."""
function adjacency_list(net::NetworkData)
    adj = Dict{String, Vector{Tuple{String, Float64}}}()
    for n in net.all_nodes
        adj[n] = Tuple{String, Float64}[]
    end
    for (a, b) in net.edges
        g = net.conductances[(a, b)]
        push!(adj[a], (b, g))
        push!(adj[b], (a, g))
    end
    return adj
end

# ─── Model constants ─────────────────────────────────────────────────────────

const V_MIN = 0.8
const V_MAX = 1.2
const I_GEN_MIN = 0.0
const I_GEN_MAX = 1.0
const I_DEM_MIN = -1.0
const I_DEM_MAX = 0.0
const P_MAX = 1.5

# ─── NLP reference model (direct bilinear constraints via Ipopt) ─────────────

function build_nlp_model(net::NetworkData)
    model = JuMP.Model()
    adj = adjacency_list(net)

    # Variables
    @variable(model, V_MIN <= V[n in net.all_nodes] <= V_MAX)
    @variable(model, I_GEN_MIN <= Ig[g in net.gen_nodes] <= I_GEN_MAX)
    @variable(model, I_DEM_MIN <= Id[d in net.dem_nodes] <= I_DEM_MAX)
    @variable(model, 0.0 <= Pg[g in net.gen_nodes] <= P_MAX)
    @variable(model,
        0.0 <= delta[g in net.gen_nodes, k in 1:(net.K)] <= net.segment_widths[g][k])

    # Objective: min Σ m_{i,k} · δ_{i,k}
    @objective(model, Min, sum(
        net.marginal_costs[g][k] * delta[g, k]
        for g in net.gen_nodes, k in 1:(net.K)
    ))

    # Bilinear: Pg = V · I (generators)
    @constraint(model, [g in net.gen_nodes], Pg[g] == V[g] * Ig[g])

    # Bilinear: V · I = -d (demands)
    @constraint(model, [d in net.dem_nodes], V[d] * Id[d] == -net.demands[d])

    # KCL
    for g in net.gen_nodes
        @constraint(model, Ig[g] == sum(c * (V[g] - V[j]) for (j, c) in adj[g]))
    end
    for d in net.dem_nodes
        @constraint(model, Id[d] == sum(c * (V[d] - V[j]) for (j, c) in adj[d]))
    end

    # Delta link: Pg = Σ δ_{g,k}
    @constraint(model, [g in net.gen_nodes],
        Pg[g] == sum(delta[g, k] for k in 1:(net.K)))

    return model
end

# ─── IOM container setup ─────────────────────────────────────────────────────

function make_container()
    sys = BenchmarkSystem(100.0)
    settings = IOM.Settings(sys; horizon = Dates.Hour(1), resolution = Dates.Hour(1))
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, 1:1)
    return container
end

# ─── MIP model using IOM bilinear approximations ─────────────────────────────

bilinear_methods = (
    ("Bin2+sSOS", IOM._add_sos2_bilinear_approx!, ()),
    ("Bin2+mSOS+McQuad", IOM._add_manual_sos2_bilinear_approx!, (add_mccormick = true,)),
    ("Bin2+Saw", IOM._add_sawtooth_bilinear_approx!, ()),
    ("Bin2+DNMDT", IOM._add_dmndt_bilinear_approx!, (double = true,)),
    ("Bin2+T-DNMDT", IOM._add_dmndt_bilinear_approx!, (double = true, tighten = true,)),
    ("Bin2+DNMDT", IOM._add_dmndt_bilinear_approx!, (double = true,)),
    ("Bin2+DNMDT+McQuad", IOM._add_dmndt_bilinear_approx!, (double = true, add_mccormick = true)),
    ("HybS+sSOS", IOM._add_hybs_sos2_bilinear_approx!, ()),
    ("HybS+sSOS+McAll", IOM._add_hybs_sos2_bilinear_approx!, (add_mccormick = true, add_quad_mccormick = true)),
    ("HybS+Saw", IOM._add_hybs_sawtooth_bilinear_approx!, ()),
    ("HybS+Saw+McAll", IOM._add_hybs_sawtooth_bilinear_approx!, (add_mccormick = true, add_quad_mccormick = true)),
    ("HybS+T-Saw", IOM._add_hybs_sawtooth_bilinear_approx!, (tighten = true,)),
    ("HybS+T-Saw+McBil", IOM._add_hybs_sawtooth_bilinear_approx!, (tighten = true, add_mccormick = true)),
    ("NMDT", IOM._add_dmndt_bilinear_approx!, ()),
    ("DNMDT", IOM._add_dmndt_bilinear_approx!, (double = true,)),
    ("DNMDT+McBil", IOM._add_dmndt_bilinear_approx!, (double = true, add_mccormick = true))
)

"""
    build_mip_model(net, method, refinement) -> NamedTuple

Build the MIP approximation of the bilinear delta model.

# Arguments
- `net::NetworkData`: network data
- `method::Symbol`: `:sos2`, `:manual_sos2`, or `:sawtooth`
- `refinement::Int`: number of PWL segments (SOS2) or sawtooth depth
"""
function build_mip_model(net::NetworkData, bilinear_fn!, bilinear_kwargs, refinement::Int)
    container = make_container()
    jump_model = IOM.get_jump_model(container)
    time_steps = 1:1
    adj = adjacency_list(net)

    # --- V and I variable containers via IOM ---
    V_container = IOM.add_variable_container!(
        container, VoltageVariable(), NetworkNode,
        net.all_nodes, time_steps,
    )
    I_container = IOM.add_variable_container!(
        container, CurrentVariable(), NetworkNode,
        net.all_nodes, time_steps,
    )

    for n in net.all_nodes
        V_container[n, 1] = JuMP.@variable(
            jump_model, base_name = "V_$(n)",
            lower_bound = V_MIN, upper_bound = V_MAX,
        )
    end
    for g in net.gen_nodes
        I_container[g, 1] = JuMP.@variable(
            jump_model, base_name = "I_$(g)",
            lower_bound = I_GEN_MIN, upper_bound = I_GEN_MAX,
        )
    end
    for d in net.dem_nodes
        I_container[d, 1] = JuMP.@variable(
            jump_model, base_name = "I_$(d)",
            lower_bound = I_DEM_MIN, upper_bound = I_DEM_MAX,
        )
    end

    # Generator bilinear: z_gen ≈ V · I_gen
    z_gen = bilinear_fn!(
        container, NetworkNode, net.gen_nodes, time_steps,
        V_container, I_container, 
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        refinement, "gen"; bilinear_kwargs...
    )

    # Demand bilinear: z_dem ≈ V · I_dem
    z_dem = bilinear_fn!(
        container, NetworkNode, net.dem_nodes, time_steps,
        V_container, I_container, 
        V_MIN, V_MAX, I_DEM_MIN, I_DEM_MAX,
        refinement, "dem"; bilinear_kwargs...
    )

    # --- Remaining linear model components (directly in JuMP) ---
    Pg = Dict{String, JuMP.VariableRef}()
    delta = Dict{Tuple{String, Int}, JuMP.VariableRef}()

    for g in net.gen_nodes
        Pg[g] = JuMP.@variable(jump_model, base_name = "Pg_$(g)",
            lower_bound = 0.0, upper_bound = P_MAX)
        for k in 1:(net.K)
            delta[g, k] = JuMP.@variable(jump_model, base_name = "delta_$(g)_$(k)",
                lower_bound = 0.0, upper_bound = net.segment_widths[g][k])
        end
    end

    # Objective: min Σ m_{i,k} · δ_{i,k}
    JuMP.@objective(jump_model, Min, sum(
        net.marginal_costs[g][k] * delta[g, k]
        for g in net.gen_nodes, k in 1:(net.K)
    ))

    # Pg == bilinear approx of V·I (generators)
    for g in net.gen_nodes
        JuMP.@constraint(jump_model, Pg[g] == z_gen[g, 1])
    end

    # V·I == -d (demands)
    for d in net.dem_nodes
        JuMP.@constraint(jump_model, z_dem[d, 1] == -net.demands[d])
    end

    # KCL: I_i = Σ g_{ij}(V_i - V_j)
    for n in net.all_nodes
        JuMP.@constraint(jump_model,
            I_container[n, 1] == sum(
                c * (V_container[n, 1] - V_container[j, 1])
                for (j, c) in adj[n]
            ))
    end

    # Delta link: Pg = Σ δ_{g,k}
    for g in net.gen_nodes
        JuMP.@constraint(jump_model,
            Pg[g] == sum(delta[g, k] for k in 1:(net.K)))
    end

    return (; container, jump_model, V_container, I_container, z_gen, z_dem)
end

# ─── Metrics ──────────────────────────────────────────────────────────────────

function compute_bilinear_residuals(result, net)
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
    geometric_mean = reduce(*, residuals)^(1/length(residuals))
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
function run_benchmark(;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
    segment_counts::Vector{Int} = [2],#, 4, 8],
    sawtooth_depths::Vector{Int} = [2]#, 4, 8],
)
    net = generate_network(; N, K, seed)
    println("Network: $(net.N) nodes, $(length(net.edges)) edges, $(net.K) cost segments")
    println("Generators: $(length(net.gen_nodes)), Demands: $(length(net.dem_nodes))")
    println()

    # --- NLP reference ---
    nlp_obj = NaN
    if HAS_IPOPT
        println("=" ^ 100)
        println("NLP Reference (Ipopt, bilinear V·I constraints)")
        println("=" ^ 100)
        nlp_model = build_nlp_model(net)
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
    println("=" ^ 100)
    println("MIP Bilinear Approximations (HiGHS)")
    println("  Refinement = num_segments for SOS2 methods, depth for Sawtooth")
    println("=" ^ 100)
    @printf("%-15s %4s %6s %7s %6s %12s %9s %11s %10s %8s\n",
        "Method", "Ref", "Vars", "Constrs", "Bins", "Objective", "Gap(%)", "Mean Resid", "Max Resid", "Time(s)")
    println("-" ^ 100)

    refinements = [2]
    for (label, fn, kw) in bilinear_methods, ref in refinements
        build_t = @elapsed begin
            result = build_mip_model(net, fn, kw, ref)
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
            gap = isnan(nlp_obj) ? NaN : abs(nlp_obj - obj) / max(abs(nlp_obj), 1e-10) * 100.0
            geometric_mean, max = compute_bilinear_residuals(result, net)
            gap_str = isnan(gap) ? "    -" : @sprintf("%8.4f", gap)
            @printf("%-15s %4d %6d %7d %6d %12.6f %8s %11.2e %10.2e %8.4f\n",
                label, ref,
                sz.variables, sz.constraints, sz.binaries,
                obj, gap_str, geometric_mean, max, solve_t)
        else
            @printf("%-15s %4d %6d %7d %6d %12s %9s %10s %8.4f\n",
                label, ref,
                sz.variables, sz.constraints, sz.binaries,
                string(status), "-", "-", solve_t)
        end
        println()
    end
    println("=" ^ 100)
end

# ─── Entry point ──────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    N = get(ARGS, 1, "10") |> x -> parse(Int, x)
    K = get(ARGS, 2, "3") |> x -> parse(Int, x)
    seed = get(ARGS, 3, "42") |> x -> parse(Int, x)
    run_benchmark(; N, K, seed)
end
