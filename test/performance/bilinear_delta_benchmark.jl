"""
Benchmark script for the bilinear delta model described in `bilinear_delta_model.tex`.

Builds a network OPF with delta (incremental) PWL costs and bilinear V·I power
balance constraints. Compares bilinear approximation methods from
InfrastructureOptimizationModels against an exact NLP reference (Ipopt).

Supports two network problem types via multiple dispatch:
  - `MockLosslessNetworkProblem`: P = V·I
  - `MockLossyNetworkProblem`:    P = V·I − a·I² − b·I − c

Usage:
    julia --project=test scripts/bilinear_delta_benchmark_github.jl [N] [K] [seed] [--lossy]

    N       = number of nodes (default 10, must be even)
    K       = number of PWL cost segments per generator (default 3)
    seed    = random seed for network generation (default 42)
    --lossy = also run the lossy generator benchmark
    --build-only = don't solve
"""

using InfrastructureOptimizationModels
using JuMP
using HiGHS
using Dates
using Random
using Printf

const IOM = InfrastructureOptimizationModels
const IS = IOM.IS

include("../mocks/mock_system.jl")
include("../mocks/mock_components.jl")

abstract type MockAbstractNetworkProblem end

struct MockLosslessNetworkProblem <: MockAbstractNetworkProblem
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

struct MockLossyNetworkProblem <: MockAbstractNetworkProblem
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
    loss::Dict{String, Vector{Float64}}
end

# ─── Network generation ──────────────────────────────────────────────────────

"""
Generate the base network topology, demands, and PWL cost data shared by all
problem types. Returns a NamedTuple consumed by the dispatched constructors.
"""
function _generate_base_network(rng, N::Int, K::Int)
    all_nodes = ["n$(i)" for i in 1:N]
    gen_nodes = all_nodes[1:(N ÷ 2)]
    dem_nodes = all_nodes[(N ÷ 2 + 1):N]

    edges = Tuple{String, String}[]
    conductances = Dict{Tuple{String, String}, Float64}()

    # Random spanning tree
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

    demands = Dict(d => 0.05 + 0.1 * rand(rng) for d in dem_nodes)

    marginal_costs = Dict{String, Vector{Float64}}()
    segment_widths = Dict{String, Vector{Float64}}()
    for g in gen_nodes
        mc = sort(rand(rng, K) .* 10.0 .+ 1.0)
        marginal_costs[g] = mc
        widths = rand(rng, K) .+ 0.1
        widths .*= 1.5 / sum(widths)
        segment_widths[g] = widths
    end

    return (;
        all_nodes, gen_nodes, dem_nodes,
        edges, conductances, demands,
        marginal_costs, segment_widths,
    )
end

function generate_network(
    ::Type{MockLosslessNetworkProblem};
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
)
    @assert iseven(N) "N must be even"
    rng = MersenneTwister(seed)
    b = _generate_base_network(rng, N, K)
    return MockLosslessNetworkProblem(
        N, K, b.gen_nodes, b.dem_nodes, b.all_nodes,
        b.edges, b.conductances, b.demands,
        b.marginal_costs, b.segment_widths,
    )
end

"""
Generate a lossy network. Same base topology as the lossless variant, with
additional per-generator loss coefficients a, b, c ∈ [0, 0.01].
"""
function generate_network(
    ::Type{MockLossyNetworkProblem};
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
)
    @assert iseven(N) "N must be even"
    rng = MersenneTwister(seed)
    b = _generate_base_network(rng, N, K)
    loss = Dict(g => 0.01 * rand(rng, 3) for g in b.gen_nodes)
    return MockLossyNetworkProblem(
        N, K, b.gen_nodes, b.dem_nodes, b.all_nodes,
        b.edges, b.conductances, b.demands,
        b.marginal_costs, b.segment_widths,
        loss,
    )
end

"""Build adjacency list from edge set."""
function adjacency_list(net::MockAbstractNetworkProblem)
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

# ─── Exact bilinear / quadratic (NLP via Ipopt) ─────────────────────────────

"""
Exact bilinear product z = x·y as a quadratic constraint.
Same calling convention as IOM bilinear approximation functions.
"""
function _add_exact_bilinear!(
    container, ::Type{C}, names, time_steps,
    x_var, y_var, x_min, x_max, y_min, y_max,
    _refinement, meta; _kwargs...,
) where {C}
    jump_model = IOM.get_jump_model(container)
    z_lo = min(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    z_hi = max(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    z = Dict{Tuple{String, Int}, JuMP.VariableRef}()
    for name in names, t in time_steps
        z_var = JuMP.@variable(jump_model, base_name = "z_exact_$(meta)_$(name)",
            lower_bound = z_lo, upper_bound = z_hi)
        JuMP.@constraint(jump_model, z_var == x_var[name, t] * y_var[name, t])
        z[(name, t)] = z_var
    end
    return z
end

"""
Exact quadratic z = x² as a quadratic constraint.
Same calling convention as IOM quadratic approximation functions.
"""
function _add_exact_quadratic!(
    container, ::Type{C}, names, time_steps,
    x_var, x_min, x_max,
    _refinement, _meta; _kwargs...,
) where {C}
    jump_model = IOM.get_jump_model(container)
    sq_lo = min(x_min^2, x_max^2)
    sq_hi = max(x_min^2, x_max^2)
    z = Dict{Tuple{String, Int}, JuMP.VariableRef}()
    for name in names, t in time_steps
        z_var = JuMP.@variable(jump_model, base_name = "z_exact_sq_$(name)",
            lower_bound = sq_lo, upper_bound = sq_hi)
        JuMP.@constraint(jump_model, z_var == x_var[name, t]^2)
        z[(name, t)] = z_var
    end
    return z
end

# ─── IOM container setup ─────────────────────────────────────────────────────

function make_container()
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(1),
        resolution = Dates.Hour(1),
        warm_start = false,
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, 1:1)
    return container
end

# ─── Dispatched model components ─────────────────────────────────────────────

function add_gen_power_constraints!(
    cons_container, jump_model, ::MockLosslessNetworkProblem, gen_nodes, z_gen,
    _I_container, _I_sq, Pg,
)
    for g in gen_nodes
        cons_container[g, 1] = JuMP.@constraint(jump_model, Pg[g, 1] == z_gen[g, 1])
    end
end

function add_gen_power_constraints!(
    cons_container, jump_model, net::MockLossyNetworkProblem, gen_nodes, z_gen, I_container,
    I_sq, Pg,
)
    for g in gen_nodes
        cons_container[g, 1] = JuMP.@constraint(jump_model,
            Pg[g, 1] ==
            z_gen[g, 1]
            - net.loss[g][1] * I_sq[g, 1]
            - net.loss[g][2] * I_container[g, 1]
            -
            net.loss[g][3])
    end
end

# ─── Method family types for dispatch ─────────────────────────────────────────

abstract type MethodFamily end
struct SeparableMethod <: MethodFamily end
struct DNMDTMethod <: MethodFamily end
struct ExactMethod <: MethodFamily end

# ─── Dispatched gen bilinear construction ─────────────────────────────────────

"""
Lossless networks: no precomputation, call the bilinear wrapper directly.
"""
function build_gen_bilinear(
    container, net::MockLosslessNetworkProblem, V_container, I_container, time_steps,
    ::MethodFamily, bilinear_fn!, bilinear_kwargs, refinement, quad_fn!,
)
    z_gen = bilinear_fn!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        refinement, "gen"; bilinear_kwargs...,
    )
    return z_gen, nothing
end

"""
Lossy + separable methods: precompute V² and I², call wrapper's precomputed overload.
I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockLossyNetworkProblem, V_container, I_container, time_steps,
    ::SeparableMethod, bilinear_fn!, bilinear_kwargs, refinement, quad_fn!,
)
    V_sq = quad_fn!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, V_MIN, V_MAX, refinement, "gen_x",
    )
    I_sq = quad_fn!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_container, I_GEN_MIN, I_GEN_MAX, refinement, "gen_y",
    )
    z_gen = bilinear_fn!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_sq, I_sq,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        V_sq, I_sq, refinement, "gen"; bilinear_kwargs...,
    )
    return z_gen, I_sq
end

"""
Lossy + DNMDT: discretize V and I, compute I² from I's discretization,
call DNMDT bilinear with pre-built discretizations. I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockLossyNetworkProblem, V_container, I_container, time_steps,
    ::DNMDTMethod, bilinear_fn!, bilinear_kwargs, refinement, quad_fn!,
)
    V_disc = IOM._discretize!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, V_MIN, V_MAX, refinement, "gen_V",
    )
    I_disc = IOM._discretize!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_container, I_GEN_MIN, I_GEN_MAX, refinement, "gen_I",
    )
    I_sq = IOM._add_dnmdt_quadratic_approx!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_disc, "gen_I_sq",
    )
    z_gen = IOM._add_dnmdt_bilinear_approx!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_disc, I_disc, "gen",
    )
    return z_gen, I_sq
end

"""
Lossy + exact (NLP): exact bilinear V·I and exact quadratic I².
"""
function build_gen_bilinear(
    container, net::MockLossyNetworkProblem, V_container, I_container, time_steps,
    ::ExactMethod, bilinear_fn!, bilinear_kwargs, refinement, quad_fn!,
)
    z_gen = _add_exact_bilinear!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        refinement, "gen"; bilinear_kwargs...,
    )
    I_sq = _add_exact_quadratic!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_container, I_GEN_MIN, I_GEN_MAX,
        refinement, "gen_I_sq",
    )
    return z_gen, I_sq
end

# ─── MIP model (unified) ─────────────────────────────────────────────────────

"""
    build_mip_model(net, method_family, bilinear_fn!, bilinear_kwargs, refinement; quad_fn!) -> NamedTuple

Build the MIP (or NLP) model for any `AbstractNetworkProblem`. Generator power
constraints and bilinear precomputation are dispatched on the network type and
method family.
"""
function build_mip_model(
    net::MockAbstractNetworkProblem, method_family::MethodFamily,
    bilinear_fn!, bilinear_kwargs, refinement::Int;
    quad_fn! = IOM._add_sos2_quadratic_approx!,
)
    container = make_container()
    jump_model = IOM.get_jump_model(container)
    time_steps = 1:1
    adj = adjacency_list(net)

    gen_devices = [MockNetworkNode(g, true) for g in net.gen_nodes]
    dem_devices = [MockNetworkNode(d, false) for d in net.dem_nodes]
    all_devices = [gen_devices; dem_devices]

    IOM.add_variables!(container, ActivePowerVariable, gen_devices, nothing)
    IOM.add_variables!(container, MockVoltageVariable, all_devices, nothing)
    IOM.add_variables!(container, MockCurrentVariable, all_devices, nothing)

    V_container = IOM.get_variable(container, MockVoltageVariable(), MockNetworkNode)
    I_container = IOM.get_variable(container, MockCurrentVariable(), MockNetworkNode)
    Pg = IOM.get_variable(container, ActivePowerVariable(), MockNetworkNode)

    # --- Bilinear gen: dispatched on network type and method family ---
    z_gen, I_sq = build_gen_bilinear(
        container, net, V_container, I_container, time_steps,
        method_family, bilinear_fn!, bilinear_kwargs, refinement, quad_fn!,
    )

    # --- Bilinear dem: always uses the wrapper (no precomputation needed) ---
    z_dem = bilinear_fn!(
        container, MockNetworkNode, net.dem_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_DEM_MIN, I_DEM_MAX,
        refinement, "dem"; bilinear_kwargs...,
    )

    pwl_link_constraints = IOM.add_constraints_container!(
        container,
        IOM.PiecewiseLinearBlockIncrementalOfferConstraint(),
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
    )
    for g in net.gen_nodes
        breakpoints = vcat(0.0, cumsum(net.segment_widths[g]))
        pwl_vars = IOM.add_pwl_variables_delta!(
            container,
            IOM.PiecewiseLinearBlockIncrementalOffer,
            MockNetworkNode,
            g,
            1,
            net.K;
            upper_bound = Inf,
        )
        IOM.add_pwl_block_offer_constraints!(
            jump_model,
            pwl_link_constraints,
            g,
            1,
            Pg[g, 1],
            pwl_vars,
            breakpoints,
        )

        pwl_cost = IOM.get_pwl_cost_expression_delta(
            pwl_vars,
            net.marginal_costs[g],
            1.0,
        )
        IOM.add_to_objective_invariant_expression!(container, pwl_cost)
    end

    # Objective: min Σ m_{i,k} · δ_{i,k}, assembled via IOM's delta-PWL helper.
    # With a 1-hour benchmark resolution, the formulation multiplier is dt = 1.0.
    IOM.update_objective_function!(container)

    # --- Generator power (dispatched) ---
    gen_pwr_constraints = IOM.add_constraints_container!(
        container,
        MockPowerEqualityConstraint(),
        MockNetworkNode,
        net.gen_nodes,
        time_steps;
        meta = "Pg",
    )
    add_gen_power_constraints!(
        gen_pwr_constraints,
        jump_model,
        net,
        net.gen_nodes,
        z_gen,
        I_container,
        I_sq,
        Pg,
    )

    # --- Demand: V·I == -d ---
    dem_pwr_constraints = IOM.add_constraints_container!(
        container,
        MockPowerEqualityConstraint(),
        MockNetworkNode,
        net.dem_nodes,
        time_steps;
        meta = "Pd",
    )
    for d in net.dem_nodes
        dem_pwr_constraints[d, 1] = JuMP.@constraint(
            jump_model, z_dem[d, 1] == -net.demands[d]
        )
    end

    # --- KCL: I_i = Σ g_{ij}(V_i - V_j) ---
    kcl_expressions = IOM.add_expression_container!(
        container,
        MockKCLExpression(),
        MockNetworkNode,
        net.all_nodes,
        time_steps,
    )
    kcl_constraints = IOM.add_constraints_container!(
        container,
        MockKCLConstraint(),
        MockNetworkNode,
        net.all_nodes,
        time_steps,
    )
    for n in net.all_nodes
        expr = kcl_expressions[n, 1] = JuMP.AffExpr(0.0)
        for (j, c) in adj[n]
            IOM.add_proportional_to_jump_expression!(
                expr, V_container[n, 1], c,
            )
            IOM.add_proportional_to_jump_expression!(
                expr, V_container[j, 1], -c,
            )
        end
        kcl_constraints[n, 1] = JuMP.@constraint(
            jump_model, I_container[n, 1] == expr
        )
    end

    return (; container, jump_model, V_container, I_container, z_gen, z_dem, I_sq)
end

# Try to load Ipopt for NLP reference; skip if unavailable
const HAS_IPOPT = try
    @eval using Ipopt
    true
catch
    false
end

function on_github()
    return get(ENV, "CI", "false") == "true" || haskey(ENV, "GITHUB_ACTIONS")
end

function on_hpc()
    return false
end

# ─── Metrics ──────────────────────────────────────────────────────────────────

"""
Compute per-node relative residuals |true − approx| / |true| for the bilinear
product. For lossless generators the ground truth is V·I; for lossy generators
it is V·I − a·I² − b·I − c.
"""
function compute_bilinear_residuals(result, net::MockLosslessNetworkProblem)
    V = result.V_container
    I = result.I_container
    residuals = Float64[]
    for (nodes, z) in ((net.gen_nodes, result.z_gen), (net.dem_nodes, result.z_dem))
        for n in nodes
            product = JuMP.value(V[n, 1]) * JuMP.value(I[n, 1])
            product == 0.0 && continue
            resid = abs(product - JuMP.value(z[n, 1])) / abs(product)
            push!(residuals, resid)
        end
    end
    geometric_mean = reduce(*, residuals)^(1 / length(residuals))
    return geometric_mean, maximum(residuals)
end

function compute_bilinear_residuals(result, net::MockLossyNetworkProblem)
    V = result.V_container
    I = result.I_container
    I_sq = result.I_sq
    residuals = Float64[]

    # Generator nodes: ground truth = V·I − a·I² − b·I − c
    for g in net.gen_nodes
        v = JuMP.value(V[g, 1])
        i = JuMP.value(I[g, 1])
        true_power = v * i - net.loss[g][1] * i^2 - net.loss[g][2] * i - net.loss[g][3]
        approx_power =
            JuMP.value(result.z_gen[g, 1]) -
            net.loss[g][1] * JuMP.value(I_sq[g, 1]) -
            net.loss[g][2] * i -
            net.loss[g][3]
        true_power == 0.0 && continue
        push!(residuals, abs(true_power - approx_power) / abs(true_power))
    end

    # Demand nodes: ground truth = V·I (unchanged)
    for d in net.dem_nodes
        product = JuMP.value(V[d, 1]) * JuMP.value(I[d, 1])
        product == 0.0 && continue
        resid = abs(product - JuMP.value(result.z_dem[d, 1])) / abs(product)
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

# ─── Benchmark output helpers ─────────────────────────────────────────────────

function print_network_info(net::MockLosslessNetworkProblem)
    println(
        "Lossless Network: $(net.N) nodes, $(length(net.edges)) edges, $(net.K) cost segments",
    )
    println("Generators: $(length(net.gen_nodes)), Demands: $(length(net.dem_nodes))")
end

function print_network_info(net::MockLossyNetworkProblem)
    println(
        "Lossy Network: $(net.N) nodes, $(length(net.edges)) edges, $(net.K) cost segments",
    )
    println("Generators: $(length(net.gen_nodes)), Demands: $(length(net.dem_nodes))")
    println("Loss coefficients (a, b, c) per generator:")
    for g in net.gen_nodes
        @printf("  %s: a=%.6f  b=%.6f  c=%.6f\n",
            g, net.loss[g][1], net.loss[g][2], net.loss[g][3])
    end
end

# ─── Main benchmark ──────────────────────────────────────────────────────────

"""
    run_benchmark(::Type{T}; N, K, seed) where T <: MockAbstractNetworkProblem

Run the full benchmark for the given network problem type.

If Ipopt is available, an exact NLP reference is included as the first method
row (using Ipopt to solve exact bilinear/quadratic constraints).
"""
function run_benchmark(
    ::Type{T};
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
    build_only::Bool = false,
) where {T <: MockAbstractNetworkProblem}
    net = generate_network(T; N, K, seed)
    print_network_info(net)
    println()

    # Build method list; include exact NLP reference when Ipopt is available
    all_methods = Any[bilinear_methods...]
    if HAS_IPOPT
        pushfirst!(all_methods,
            ("Exact (NLP)", ExactMethod(), _add_exact_bilinear!, (), _add_exact_quadratic!),
        )
    else
        println("Ipopt not available — skipping NLP reference.")
        println("Install Ipopt.jl in the test environment for NLP comparison.")
        println()
    end

    println("="^110)
    println("Bilinear Approximation Benchmarks")
    println("  Refinement = num_segments for SOS2 methods, depth for Sawtooth/DNMDT")
    println("="^110)
    @printf("%-17s %4s %6s %7s %6s %12s %9s %11s %10s %8s %8s\n",
        "Method", "Ref", "Vars", "Constrs", "Bins", "Objective",
        "Gap(%)", "Mean Resid", "Max Resid", "build_t", "solve_t")
    println("-"^110)

    nlp_obj = NaN
    refinements = [2, 4, 6]

    for (label, family, fn, kw, qfn) in all_methods
        is_exact = family isa ExactMethod
        refs = is_exact ? [0] : refinements

        for ref in refs
            build_t = @elapsed begin
                result = build_mip_model(net, family, fn, kw, ref; quad_fn! = qfn)
            end

            if is_exact
                JuMP.set_optimizer(result.jump_model, Ipopt.Optimizer)
                JuMP.set_optimizer_attribute(result.jump_model, "print_level", 0)
            else
                JuMP.set_optimizer(result.jump_model, HiGHS.Optimizer)
                JuMP.set_optimizer_attribute(result.jump_model, "log_to_console", false)
                JuMP.set_optimizer_attribute(result.jump_model, "time_limit", 300.0)
            end

            if !build_only
                solve_t = @elapsed JuMP.optimize!(result.jump_model)
                status = JuMP.termination_status(result.jump_model)
            else
                solve_t = 0.0
                status = nothing
            end
            sz = model_size(result.jump_model)

            solved = if is_exact
                status == JuMP.LOCALLY_SOLVED
            else
                status in (JuMP.OPTIMAL, JuMP.TIME_LIMIT) &&
                    JuMP.has_values(result.jump_model)
            end

            if solved
                obj = JuMP.objective_value(result.jump_model)
                if is_exact
                    nlp_obj = obj
                end
                gap = if isnan(nlp_obj)
                    NaN
                else
                    abs(nlp_obj - obj) / max(abs(nlp_obj), 1e-10) * 100.0
                end
                geometric_mean, max_resid = compute_bilinear_residuals(result, net)
                gap_str = isnan(gap) ? "    -" : @sprintf("%8.4f", gap)
                ref_str = is_exact ? "  -" : @sprintf("%4d", ref)
                @printf("%-17s %4s %6d %7d %6d %12.6f %8s %11.2e %10.2e %8.4f %8.4f\n",
                    label, ref_str,
                    sz.variables, sz.constraints, sz.binaries,
                    obj, gap_str, geometric_mean, max_resid, build_t, solve_t)
            else
                ref_str = is_exact ? "  -" : @sprintf("%4d", ref)
                @printf("%-17s %4s %6d %7d %6d %12s %8s %11s %10s %8.4f %8.4f\n",
                    label, ref_str,
                    sz.variables, sz.constraints, sz.binaries,
                    string(status), "-", "-", "-", build_t, solve_t)
            end
        end
        println()
    end
    println("="^110)
end

# ─── Entry point ──────────────────────────────────────────────────────────────

if on_github()
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
            IOM._add_sawtooth_quadratic_approx!,
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
            IOM._add_sawtooth_quadratic_approx!,
        ),
        (
            "DNMDT",
            DNMDTMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (),
            IOM._add_dnmdt_quadratic_approx!,
        ),
    )
elseif on_hpc()
    bilinear_methods = (
        (
            "Bin2+sSOS",
            SeparableMethod(),
            IOM._add_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
        ),
        (
            "Bin2+mSOS+McQuad",
            SeparableMethod(),
            IOM._add_manual_sos2_bilinear_approx!,
            (add_mccormick = true,),
            IOM._add_sos2_quadratic_approx!,
        ),
        (
            "Bin2+Saw",
            SeparableMethod(),
            IOM._add_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
        ),
        (
            "Bin2+DNMDT",
            SeparableMethod(),
            IOM._add_dnmdt_quadratic_bilinear_approx!,
            (double = true,),
            IOM._add_dnmdt_quadratic_approx!,
        ),
        (
            "Bin2+T-DNMDT",
            SeparableMethod(),
            IOM._add_dnmdt_quadratic_bilinear_approx!,
            (double = true, tighten = true),
            IOM._add_dnmdt_quadratic_approx!,
        ),
        (
            "Bin2+DNMDT+McQuad",
            SeparableMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (double = true, add_mccormick = true),
            IOM._add_dnmdt_quadratic_approx!,
        ),
        (
            "HybS+sSOS",
            SeparableMethod(),
            IOM._add_hybs_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
        ),
        (
            "HybS+sSOS+McAll",
            SeparableMethod(),
            IOM._add_hybs_sos2_bilinear_approx!,
            (add_mccormick = true, add_quad_mccormick = true),
            IOM._add_sos2_quadratic_approx!,
        ),
        (
            "HybS+Saw",
            SeparableMethod(),
            IOM._add_hybs_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
        ),
        (
            "HybS+Saw+McAll",
            SeparableMethod(),
            IOM._add_hybs_sawtooth_bilinear_approx!,
            (add_mccormick = true, add_quad_mccormick = true),
            IOM._add_sawtooth_quadratic_approx!,
        ),
        (
            "HybS+T-Saw",
            SeparableMethod(),
            IOM._add_hybs_sawtooth_bilinear_approx!,
            (tighten = true,),
            IOM._add_sawtooth_quadratic_approx!,
        ),
        (
            "HybS+T-Saw+McBil",
            SeparableMethod(),
            IOM._add_hybs_sawtooth_bilinear_approx!,
            (tighten = true, add_mccormick = true),
            IOM._add_sawtooth_quadratic_approx!,
        ),
        (
            "NMDT",
            DNMDTMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (double = false,),
            IOM._add_dnmdt_quadratic_approx!,
        ),
        (
            "DNMDT",
            DNMDTMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (double = true,),
            IOM._add_dnmdt_quadratic_approx!,
        ),
        (
            "DNMDT+McBil",
            DNMDTMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (double = true, add_mccormick = true),
            IOM._add_dnmdt_quadratic_approx!,
        ),
    )
else
    # Running locally
    bilinear_methods = (
        (
        "Bin2+sSOS",
        SeparableMethod(),
        IOM._add_sos2_bilinear_approx!,
        (),
        IOM._add_sos2_quadratic_approx!,
    ),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    T = "--lossy" in ARGS ? MockLossyNetworkProblem : MockLosslessNetworkProblem
    N = get(ARGS, 1, "10") |> x -> parse(Int, x)
    K = get(ARGS, 2, "3") |> x -> parse(Int, x)
    seed = get(ARGS, 3, "42") |> x -> parse(Int, x)
    build_only = "--build-only" in ARGS
    run_benchmark(T; N, K, seed, build_only)
end
