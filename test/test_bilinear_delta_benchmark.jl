using Random
using Dates
using JuMP
using HiGHS

using InfrastructureSystems
using InfrastructureOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization
const IOM = InfrastructureOptimizationModels

# ----------------- Problem generation

function _random_convex_pwl_points(n_tranches::Int, pmax::Float64, rng)
    xs = sort(rand(rng, n_tranches - 1)) .* pmax
    points = [(0.0, 0.0)]
    cumulative_cost = 0.0
    prev_x = 0.0
    slope = 5.0 + 20.0 * rand(rng)
    for x in xs
        cumulative_cost += slope * (x - prev_x)
        push!(points, (x, cumulative_cost))
        prev_x = x
        slope += 5.0 + 10.0 * rand(rng)
    end
    cumulative_cost += slope * (pmax - prev_x)
    push!(points, (pmax, cumulative_cost))
    return points
end

struct NetworkProblem
    size::Integer
    gens::UnitRange{Integer}
    dems::UnitRange{Integer}
    edges::Set{Tuple{Integer, Integer}}
    pmaxes::Vector{Float64}
    conductances::Dict{Tuple{Integer, Integer}, Float64}
    cost_functions::Vector{IS.CostCurve{IS.PiecewisePointCurve}}
end
function NetworkProblem(; size, n_cost_segments, seed)
    rng = MersenneTwister(seed)
    half = div(size, 2)

    edges = Set{Tuple{Integer, Integer}}()
    conductances = Dict{Tuple{Integer, Integer}, Float64}()
    permutation = shuffle(rng, 1:size)
    for i in 1:(size - 1)
        i, j = permutation[i], permutation[i + 1]
        e = i < j ? (i, j) : (j, i)
        push!(edges, e)
        conductances[e] = 0.005 + 0.005 * rand(rng)
    end
    for _ in 1:half
        i, j = shuffle(rng, 1:size)[1:2]
        e = i < j ? (i, j) : (j, i)
        push!(edges, e)
        conductances[e] = 0.005 + 0.005 * rand(rng)
    end

    pmaxes = rand(rng, half)
    cost_functions = Vector{IS.CostCurve{IS.PiecewisePointCurve}}()
    for i in 1:half
        points = _random_convex_pwl_points(n_cost_segments, pmaxes[i], rng)
        pwl = IS.PiecewiseLinearData(points)
        cost_curve = IS.CostCurve(IS.InputOutputCurve(pwl))
        push!(cost_functions, cost_curve)
    end

    return NetworkProblem(
        size,
        1:half,
        (half + 1):size,
        edges,
        rand(rng, half),
        conductances,
        cost_functions,
    )
end

# ----------------- Mock infrastructure

struct MockDeviceFormulation <: IOM.AbstractDeviceFormulation end
struct MockPowerModel <: IS.Optimization.AbstractPowerModel end

struct MockNetworkNode <: IS.InfrastructureSystemsComponent
    id::Integer
    adj::Vector{Tuple{Int, Float64}}
    pmax::Float64
    current_bounds::NTuple{2, Float64}
end
MockNetworkNode(id, adj, pmax) = # generators
    MockNetworkNode(
        id,
        adj,
        pmax,
        (0.0, 1.0),
    )
MockNetworkNode(id, adj) = # demands
    MockNetworkNode(
        id,
        adj,
        0.0,
        (-1.0, 0.0),
    )
IOM.get_name(n::MockNetworkNode) = string(n.id)
IOM.get_base_power(::MockNetworkNode) = 1.0
get_power_bounds(n::MockNetworkNode) = (0.0, n.pmax)
get_voltage_bounds(::MockNetworkNode) = (0.8, 1.2)
get_current_bounds(n::MockNetworkNode) = n.current_bounds

struct MockSystem <: IS.InfrastructureSystemsContainer
    nodes::Vector{MockNetworkNode}
end
IOM.get_base_power(::MockSystem) = 1.0
IOM.stores_time_series_in_memory(::MockSystem) = false
IOM.get_available_components(_, _, sys::MockSystem) = length(sys.nodes)
IOM.calculate_aux_variables!(_, ::MockSystem) =
    IOM.RunStatus.SUCCESSFULLY_FINALIZED
IOM.calculate_dual_variables!(_, ::MockSystem, _) =
    IOM.RunStatus.SUCCESSFULLY_FINALIZED
get_components(_, sys::MockSystem) = sys.nodes

# ----------------- Container types

struct MockKCLConstraint <: ConstraintType end
struct MockVoltageVariable <: VariableType end
struct MockCurrentVariable <: VariableType end

# ----------------- IOM patches

IOM.intermediate_set_units_base_system!(::MockSystem, base) = nothing
IOM.intermediate_get_forecast_initial_timestamp(::MockSystem) = DateTime(1970)

# ----------------- System / container building

function _build_adj(problem, i)
    adj = Vector{Tuple{Int, Float64}}()
    for edge in problem.edges
        if i in edge
            j = i == edge[1] ? edge[2] : edge[1]
            push!(adj, (j, problem.conductances[edge]))
        end
    end
    return adj
end

function build_system(problem)
    gen_nodes =
        [MockNetworkNode(i, _build_adj(problem, i), problem.pmaxes[i]) for i in problem.gens]
    dem_nodes =
        [MockNetworkNode(i, _build_adj(problem, i)) for i in problem.dems]
    return MockSystem([gen_nodes; dem_nodes])
end

function add_variables!(container, sys)
    jump_model = get_jump_model(container)
    nodes = get_components(MockNetworkNode, sys)
    variable_types = [ActivePowerVariable, MockVoltageVariable, MockCurrentVariable]
    bounds_fns = [get_power_bounds, get_voltage_bounds, get_current_bounds]

    for (variable_type, bounds_fn) in zip(variable_types, bounds_fns)
        variable = add_variable_container!(
            container,
            variable_type(),
            MockNetworkNode,
            get_name.(nodes),
            1:1,
        )

        for node in nodes
            name = get_name(node)
            lower_bound, upper_bound = bounds_fn(node)
            variable[name, 1] = JuMP.@variable(
                jump_model,
                base_name = "$(variable_type)_MockNetworkNode_{$(name), 1}",
                lower_bound = lower_bound, upper_bound = upper_bound
            )
        end
    end
    return
end

function add_constraints!(container, sys)
    jump_model = get_jump_model(container)
    nodes = get_components(MockNetworkNode, sys)
    voltages = IOM.get_variable(container, MockVoltageVariable(), MockNetworkNode)
    currents = IOM.get_variable(container, MockCurrentVariable(), MockNetworkNode)

    kcl_container = add_constraints_container!(
        container,
        MockKCLConstraint(),
        MockNetworkNode,
        get_name.(nodes),
        1:1,
    )
    for node in nodes
        i = string(node.id)
        I, Vi = currents[i, 1], voltages[i, 1]
        v_diff = JuMP.AffExpr(0.0)
        for (j, conductance) in node.adj
            Vj = voltages[string(j), 1]
            JuMP.add_to_expression!(v_diff, conductance, Vi)
            JuMP.add_to_expression!(v_diff, -conductance, Vj)
        end
        kcl_container[i, 1] = JuMP.@constraint(jump_model, I == v_diff)
    end

    for i in problem.gens
        IOM.add_variable_cost_to_objective!(
            container,
            ActivePowerVariable(),
            nodes[i],
            problem.cost_functions[i],
            MockDeviceFormulation(),
        )
    end
    return
end

function make_container(sys; optimizer = HiGHS.Optimizer)
    container = IOM.OptimizationContainer(
        sys,
        IOM.Settings(
            sys;
            horizon = Dates.Hour(1),
            resolution = Dates.Hour(1),
            optimizer,
        ),
        JuMP.Model(),
        IS.Deterministic,
    )
    IOM.set_time_steps!(container, 1:1)
    return container
end

# ----------------- Profiling and comparison

# -----------------

problem = NetworkProblem(; size = 10, n_cost_segments = 3, seed = 0)
sys = build_system(problem)
container = make_container(sys)
add_variables!(container, sys)
add_constraints!(container, sys)
network = NetworkModel(MockPowerModel)
init_optimization_container!(container, network, sys)
status = IOM.execute_optimizer!(container, sys)
@show status
