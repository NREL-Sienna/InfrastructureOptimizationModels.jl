"""
Benchmark script for the bilinear delta model described in `bilinear_delta_model.tex`.

Builds a lossy network OPF with delta (incremental) PWL costs and bilinear V·I power
balance constraints. Compares bilinear approximation methods from
InfrastructureOptimizationModels against an exact NLP reference (Ipopt).

The lossy generator model is P = V·I − a·I² − b·I − c.

Usage:
    julia --project=test test/performance/bilinear_delta_benchmark.jl [options]

Options:
    -N, --nodes INT          number of nodes (default 10, must be even)
    -K, --cost INT           number of PWL cost segments per generator (default 3)
    -S, --seed INT           random seed for network generation (default 0)
    -R, --refinements INT... refinement levels (default 4 6 8)
    -B, --build-only         don't solve, only build the model
"""

using InfrastructureOptimizationModels
using ArgParse
using JuMP
using Dates
using Random
using Printf

ENVIRONMENT = if get(ENV, "CI", "") == "true" || haskey(ENV, "GITHUB_ACTIONS")
    :github
elseif get(ENV, "HOSTNAME", "") == "kl1"
    :kestrel
else
    :local
end

LP_OPT = if ENVIRONMENT == :github
    @eval using HiGHS
    HiGHS.Optimizer
elseif ENVIRONMENT == :kestrel
    @eval using Xpress
    Xpress.Optimizer
else
    @eval import Xpress_jll
    ENV["XPRESS_JL_LIBRARY"] = Xpress_jll.libxprs
    @eval using Xpress
    Xpress.Optimizer
end

using Ipopt
using UnoSolver

const IOM = InfrastructureOptimizationModels
const IS = IOM.IS

include("../mocks/mock_system.jl")
include("../mocks/mock_components.jl")

struct MockPowerEqualityConstraint <: IOM.ConstraintType end
struct MockKCLConstraint <: IOM.ConstraintType end
struct MockLineLossConstraint <: IOM.ConstraintType end

struct MockKCLExpression <: IOM.ExpressionType end
struct MockLineLossExpression <: IOM.ExpressionType end

struct MockLineLossAuxVariable <: IOM.AuxVariableType end
function IOM.calculate_aux_variable_value!(
    container::OptimizationContainer,
    key::AuxVarKey{MockLineLossAuxVariable, MockNetworkNode},
    system::MockSystem
)
    cont = get_aux_variable(container, key)
    names = axes(cont, 1)
    time_steps = get_time_steps(container)

    # Isq_container = get_variable(container, BilinearProductExpression())
    # I_container = get_variable(container, QuadraticExpresion())

    for name in names, t in time_steps
        # Isq = JuMP.value(Isq_container[name, t])
        # I = JuMP.value(I_container[name, t])
        Isq = 1
        I = 1
        a, b, c = get_component(MockNetworkNode, system, name).loss
        cont[name, t] = -a * Isq - b * I - c
    end

    return
end

struct MockNetworkProblem
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
Generate a lossy network with per-generator loss coefficients a, b, c ∈ [0, 0.01].
"""
function generate_network(;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
)
    @assert iseven(N) "N must be even"
    rng = MersenneTwister(seed)
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

    loss = Dict(g => 0.1 * rand(rng, 3) for g in gen_nodes)
    # loss = Dict(g => zeros(3) for g in gen_nodes)

    return MockNetworkProblem(
        N, K, gen_nodes, dem_nodes, all_nodes,
        edges, conductances, demands,
        marginal_costs, segment_widths,
        loss,
    )
end

"""Build adjacency list from edge set."""
function adjacency_list(net::MockNetworkProblem)
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
    system = MockSystem(100.0)
    settings = IOM.Settings(
        system;
        horizon = Dates.Hour(1),
        resolution = Dates.Hour(1),
        warm_start = false,
    )
    container = IOM.OptimizationContainer(system, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, 1:1)
    return container, system
end

# ─── Method family types for dispatch ─────────────────────────────────────────

abstract type MethodFamily end
struct SeparableMethod <: MethodFamily end
struct DNMDTMethod <: MethodFamily end
abstract type ExactMethod <: MethodFamily end
struct IpoptMethod <: ExactMethod end
struct UnoMethod <: ExactMethod end

# ─── Dispatched gen bilinear construction ─────────────────────────────────────

"""
Separable methods: precompute V² and I², call wrapper's precomputed overload.
I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    ::SeparableMethod, bilin_fn!, bilin_kwargs, quad_fn!, quad_kwargs, refinement,
)
    V_sq = quad_fn!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, V_MIN, V_MAX, refinement, "gen_x"; quad_kwargs...,
    )
    I_sq = quad_fn!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_container, I_GEN_MIN, I_GEN_MAX, refinement, "gen_y"; quad_kwargs...,
    )
    z_gen = bilin_fn!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        V_sq, I_sq, refinement, "gen"; bilin_kwargs...,
    )
    return z_gen, I_sq
end

"""
DNMDT: discretize V and I, compute I² from I's discretization,
call DNMDT bilinear with pre-built discretizations. I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    ::DNMDTMethod, bilin_fn!, bilin_kwargs, quad_fn!, quad_kwargs, refinement,
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
        I_disc, "gen_I_sq"; quad_kwargs...,
    )
    z_gen = IOM._add_dnmdt_bilinear_approx!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_disc, I_disc, "gen"; bilin_kwargs...,
    )
    return z_gen, I_sq
end

"""
Exact (NLP): exact bilinear V·I and exact quadratic I².
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    ::ExactMethod, bilin_fn!, bilin_kwargs, quad_fn!, quad_kwargs, refinement,
)
    z_gen = _add_exact_bilinear!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        refinement, "gen"; bilin_kwargs...,
    )
    I_sq = _add_exact_quadratic!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_container, I_GEN_MIN, I_GEN_MAX,
        refinement, "gen_I_sq"; quad_kwargs...,
    )
    return z_gen, I_sq
end

# ─── MIP model  ──────────────────────────────────────────────────────────────

"""
    build_mip_model(net, method_family, bilin_fn!, bilin_kwargs, quad_fn!, quad_kwargs, refinement) -> NamedTuple

Build the MIP (or NLP) model for a `MockNetworkProblem`. Bilinear
precomputation is dispatched on the method family.
"""
function build_mip_model(
    net::MockNetworkProblem, method_family::MethodFamily,
    bilin_fn!, bilin_kwargs, quad_fn!, quad_kwargs, refinement::Int,
)
    container, system = make_container()
    tdf = TestDeviceFormulation()
    jump_model = IOM.get_jump_model(container)
    time_steps = 1:1
    adj = adjacency_list(net)

    gen_devices = [MockNetworkNode(g, net.loss[g]) for g in net.gen_nodes]
    dem_devices = [MockNetworkNode(d) for d in net.dem_nodes]
    all_devices = [gen_devices; dem_devices]
    for device in all_devices
        add_component!(system, device)
    end

    IOM.add_variables!(container, ActivePowerVariable, gen_devices, tdf)
    IOM.add_variables!(container, MockVoltageVariable, all_devices, tdf)
    IOM.add_variables!(container, MockCurrentVariable, all_devices, tdf)
    IOM.add_variables!(container, MockLineLossAuxVariable, gen_devices, tdf)

    dm = DeviceModel(MockNetworkNode, TestDeviceFormulation)
    add_range_constraints!(
        container,
        MockPowerRangeConstraint,
        ActivePowerVariable,
        gen_devices,
        dm,
        TestPowerModel,
    )

    V_container = IOM.get_variable(container, MockVoltageVariable(), MockNetworkNode)
    I_container = IOM.get_variable(container, MockCurrentVariable(), MockNetworkNode)
    Pg = IOM.get_variable(container, ActivePowerVariable(), MockNetworkNode)

    # --- Bilinear gen: dispatched on network type and method family ---
    z_gen, I_sq = build_gen_bilinear(
        container, net, V_container, I_container, time_steps,
        method_family, bilin_fn!, bilin_kwargs, quad_fn!, quad_kwargs, refinement,
    )

    # --- Bilinear dem: always uses the wrapper (no precomputation needed) ---
    z_dem = bilin_fn!(
        container, MockNetworkNode, net.dem_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_DEM_MIN, I_DEM_MAX,
        refinement, "dem"; bilin_kwargs...,
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

    # --- Generator power: P = V·I − loss ---
    gen_pwr_constraints = IOM.add_constraints_container!(
        container,
        MockPowerEqualityConstraint(),
        MockNetworkNode,
        net.gen_nodes,
        time_steps;
        meta = "Pg",
    )
    for g in net.gen_nodes
        a, b, c = net.loss[g]
        gen_pwr_constraints[g, 1] =
            JuMP.@constraint(
                jump_model,
                Pg[g, 1] ==
                z_gen[g, 1] - a * I_sq[g, 1] - b * I_container[g, 1] - c
            )
    end

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

    return (; container, system, jump_model, V_container, I_container, z_gen, z_dem, I_sq)
end

# ─── Metrics ──────────────────────────────────────────────────────────────────

@inline function residual(actual, measured, eps = 1e-10)
    return abs(actual - measured) / max(abs(actual), eps)
end

"""
Compute per-node relative residuals |true − approx| / |true| for the bilinear
product. For lossy generators the ground truth is V·I − a·I² − b·I − c.
"""
function compute_bilinear_residuals(result, net::MockNetworkProblem)
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
        push!(residuals, residual(true_power, approx_power))
    end

    # Demand nodes: ground truth = V·I (unchanged)
    for d in net.dem_nodes
        product = JuMP.value(V[d, 1]) * JuMP.value(I[d, 1])
        approx = JuMP.value(result.z_dem[d, 1])
        push!(residuals, residual(product, approx))
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

function print_network_info(net::MockNetworkProblem)
    println(
        "Network: $(net.N) nodes, $(length(net.edges)) edges, $(net.K) cost segments",
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
    run_benchmark(methods, lp_opt; N, K, seed, refinements, build_only)

Run the full benchmark for the lossy network problem.

An exact NLP reference is included as the first method row
(using Ipopt/Uno to solve exact bilinear/quadratic constraints).
"""
function run_benchmark(;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
    refinements::Vector{Int} = [4, 6, 8],
    build_only::Bool = false,
)
    net = generate_network(; N, K, seed)
    print_network_info(net)
    println()

    all_methods = [
        ("NLP (Ipopt)", IpoptMethod(), _add_exact_bilinear!, (), _add_exact_quadratic!, ()),
        ("NLP (Uno)", UnoMethod(), _add_exact_bilinear!, (), _add_exact_quadratic!, ()),
        bilinear_methods[ENVIRONMENT]...,
    ]

    println("="^110)
    println("Bilinear Approximation Benchmarks")
    println("  Refinement = num_segments for SOS2 methods, depth for Sawtooth/DNMDT")
    println("="^110)
    @printf("%-17s %4s %6s %7s %6s %12s %9s %11s %10s %8s %8s\n",
        "Method", "Ref", "Vars", "Constrs", "Bins", "Objective",
        "Gap(%)", "Mean Resid", "Max Resid", "build_t", "solve_t")
    println("-"^110)

    nlp_obj = NaN

    for (label, family, fn, kw, qfn, qkw) in all_methods
        is_exact = family isa ExactMethod
        refs = is_exact ? [0] : refinements

        for ref in refs
            build_t = @elapsed begin
                result = build_mip_model(net, family, fn, kw, qfn, qkw, ref)
            end

            opt = if family isa IpoptMethod
                Ipopt.Optimizer
            elseif family isa UnoMethod
                () -> UnoSolver.Optimizer(; preset = "filtersqp")
            else
                LP_OPT
            end
            JuMP.set_optimizer(result.jump_model, opt)
            JuMP.set_silent(result.jump_model)

            if build_only
                solve_t = 0.0
                status = nothing
            else
                # solve_t = @elapsed JuMP.optimize!(result.jump_model)
                solve_t = @elapsed IOM.execute_optimizer!(result.container, result.system)
                status = JuMP.termination_status(result.jump_model)
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
                    residual(nlp_obj, obj) * 100.0
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

bilinear_methods = Dict(
    :local => (
        (
            "Bin2+sSOS",
            SeparableMethod(),
            IOM._add_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
            (),
        ),
        (
            "Bin2+mSOS",
            SeparableMethod(),
            IOM._add_manual_sos2_bilinear_approx!,
            (),
            IOM._add_manual_sos2_quadratic_approx!,
            (),
        ),
        (
            "Bin2+Saw",
            SeparableMethod(),
            IOM._add_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
            (),
        ),
        (
            "HybS+sSOS",
            SeparableMethod(),
            IOM._add_hybs_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
            (),
        ),
        (
            "HybS+mSOS",
            SeparableMethod(),
            IOM._add_hybs_manual_sos2_bilinear_approx!,
            (),
            IOM._add_manual_sos2_quadratic_approx!,
            (),
        ),
        (
            "HybS+Saw",
            SeparableMethod(),
            IOM._add_hybs_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
            (),
        ),
        (
            "DNMDT",
            DNMDTMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (),
            IOM._add_dnmdt_quadratic_approx!,
            (),
        ),
    ),
    :github => (
        (
            "Bin2+sSOS",
            SeparableMethod(),
            IOM._add_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
            (),
        ),
        (
            "Bin2+Saw",
            SeparableMethod(),
            IOM._add_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
            (),
        ),
        (
            "HybS+sSOS",
            SeparableMethod(),
            IOM._add_hybs_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
            (),
        ),
        (
            "HybS+Saw",
            SeparableMethod(),
            IOM._add_hybs_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
            (),
        ),
        (
            "DNMDT",
            DNMDTMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (),
            IOM._add_dnmdt_quadratic_approx!,
            (),
        ),
    ),
    :kestrel => (
        (
            "Bin2+sSOS",
            SeparableMethod(),
            IOM._add_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
            (),
        ),
        (
            "Bin2+mSOS",
            SeparableMethod(),
            IOM._add_manual_sos2_bilinear_approx!,
            (),
            IOM._add_manual_sos2_quadratic_approx!,
            (),
        ),
        (
            "Bin2+Saw",
            SeparableMethod(),
            IOM._add_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
            (),
        ),
        (
            "HybS+sSOS",
            SeparableMethod(),
            IOM._add_hybs_sos2_bilinear_approx!,
            (),
            IOM._add_sos2_quadratic_approx!,
            (),
        ),
        (
            "HybS+mSOS",
            SeparableMethod(),
            IOM._add_hybs_manual_sos2_bilinear_approx!,
            (),
            IOM._add_manual_sos2_quadratic_approx!,
            (),
        ),
        (
            "HybS+Saw",
            SeparableMethod(),
            IOM._add_hybs_sawtooth_bilinear_approx!,
            (),
            IOM._add_sawtooth_quadratic_approx!,
            (),
        ),
        (
            "DNMDT",
            DNMDTMethod(),
            IOM._add_dnmdt_bilinear_approx!,
            (),
            IOM._add_dnmdt_quadratic_approx!,
            (),
        ),
    ),
)

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--nodes", "-N"
        arg_type = Int
        default = 10
        help = "number of nodes (must be even)"
        "--cost", "-K"
        arg_type = Int
        default = 3
        help = "number of PWL cost segments per generator"
        "--seed", "-S"
        arg_type = Int
        default = 42
        help = "random seed for network generation"
        "--build-only", "-B"
        action = :store_true
        help = "don't solve, only build the model"
        "--refinements", "-R"
        arg_type = Int
        nargs = '+'
        default = [4, 6, 8]
        help = "refinement levels (list of integers)"
        "--github", "-G"
        action = :store_true
        help = "for github ci/cd"
        "--kestrel", "-E"
        action = :store_true
        help = "for kestrel"
    end
    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    parsed = parse_commandline()
    N = parsed["nodes"]
    K = parsed["cost"]
    seed = parsed["seed"]
    build_only = parsed["build-only"]
    refinements = parsed["refinements"]

    # Run small network so second run is faster.
    redirect_stdout(devnull) do
        run_benchmark(;
            N = 2,
            K = 1,
            seed = 0,
            build_only = false,
            refinements = [1],
        )
    end
    run_benchmark(; N, K, seed, build_only, refinements)
end
