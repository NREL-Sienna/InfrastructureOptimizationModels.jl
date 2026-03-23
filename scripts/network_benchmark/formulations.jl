# ─── Model constants ─────────────────────────────────────────────────────────

const V_MIN = 0.8
const V_MAX = 1.2
const I_GEN_MIN = 0.0
const I_GEN_MAX = 1.0
const I_DEM_MIN = -1.0
const I_DEM_MAX = 0.0
const P_MAX = 1.5

function build_model_base(net::AbstractNetworkProblem)
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

    # Objective: min Σ m_{i,k} · δ_{i,k}
    JuMP.@objective(
        jump_model,
        Min,
        sum(
            net.marginal_costs[g][k] * delta[g, k]
            for g in net.gen_nodes, k in 1:(net.K)
        )
    )

    return container, Pg, V_container, I_container
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

# ─── IOM container setup ─────────────────────────────────────────────────────

function make_container()
    sys = BenchmarkSystem(100.0)
    settings = IOM.Settings(sys; horizon = Dates.Hour(1), resolution = Dates.Hour(1))
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, 1:1)
    return container
end

# ─── VI and V^2/R formulations ───────────────────────────────────────────────

function build_vi_formulation(
    net::LosslessNetworkProblem,
    bilinear_fn!,
    bilinear_kwargs,
    refinement::Int,
)
    container, Pg, V_container, I_container = build_model_base(net)
    time_steps = 1:1
    jump_model = get_jump_model(container)

    # Generator bilinear: z_gen ≈ V · I_gen
    z_gen = bilinear_fn!(
        container, NetworkNode, net.gen_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        refinement, "gen"; bilinear_kwargs...,
    )

    # Demand bilinear: z_dem ≈ V · I_dem
    z_dem = bilinear_fn!(
        container, NetworkNode, net.dem_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_DEM_MIN, I_DEM_MAX,
        refinement, "dem"; bilinear_kwargs...,
    )

    # Pg == bilinear approx of V·I (generators)
    for g in net.gen_nodes
        JuMP.@constraint(jump_model, Pg[g] == z_gen[g, 1])
    end

    # V·I == -d (demands)
    for d in net.dem_nodes
        JuMP.@constraint(jump_model, -net.demands[d] == z_dem[d, 1])
    end

    return (; container, jump_model, V_container, I_container, z_gen, z_dem)
end

function build_vi_formulation(net::LossyNetworkProblem, bilinear_fn!, bilinear_kwargs, refinement::Int)
    container, Pg, V_container, I_container = build_model_base(net)
    time_steps = 1:1
    jump_model = get_jump_model(container)

    Vsq = IOM._add_sos2_quadratic_approx!(
        container, NetworkNode, net.all_nodes, time_steps,
        V_container, V_MIN, V_MAX, 4, "vsq",
    )
    Isq_gen = IOM._add_sos2_quadratic_approx!(
        container, NetworkNode, net.gen_nodes, time_steps,
        I_container, I_GEN_MIN, I_GEN_MAX, 4, "isq_gen",
    )
    Isq_dem = IOM._add_sos2_quadratic_approx!(
        container, NetworkNode, net.dem_nodes, time_steps,
        I_container, I_DEM_MIN, I_DEM_MAX, 4, "isq_dem",
    )

    VI_gen = IOM._add_bin2_bilinear_approx_impl!(
        container, NetworkNode, net.gen_nodes, time_steps,
        Vsq, Isq_gen, V_container, I_container,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        IOM._add_sos2_quadratic_approx!, 4, "vi_gen",
    )
    VI_dem = IOM._add_bin2_bilinear_approx_impl!(
        container, NetworkNode, net.dem_nodes, time_steps,
        Vsq, Isq_dem, V_container, I_container,
        V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX,
        IOM._add_sos2_quadratic_approx!, 4, "vi_dem",
    )

    for g in net.gen_nodes
        a, b, c = net.loss_coeffs[g]
        JuMP.@constraint(
            jump_model,
            Pg[g] ==
            VI_gen[g, 1]
                - Isq_gen[g, 1] * a
                - I_container[g, 1] * b
                - c
        )
    end

    for d in net.dem_nodes
        JuMP.@constraint(jump_model, -net.demands[d] == VI_dem[d, 1])
    end

    return (; container, jump_model, V_container, I_container, VI_gen, VI_dem)
end
