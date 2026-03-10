using Random
using Dates
using JuMP
using HiGHS

using InfrastructureSystems
using InfrastructureOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization
const IOM = InfrastructureOptimizationModels

const TEST_DIR = @__DIR__
include(joinpath(TEST_DIR, "mocks/mock_optimizer.jl"))
include(joinpath(TEST_DIR, "mocks/mock_system.jl"))
include(joinpath(TEST_DIR, "mocks/mock_components.jl"))
include(joinpath(TEST_DIR, "mocks/mock_time_series.jl"))
include(joinpath(TEST_DIR, "mocks/mock_services.jl"))
include(joinpath(TEST_DIR, "mocks/mock_container.jl"))
include(joinpath(TEST_DIR, "mocks/constructors.jl"))
include(joinpath(TEST_DIR, "test_utils/test_types.jl"))
include(joinpath(TEST_DIR, "test_utils/objective_function_helpers.jl"))

# ----------------- Mock infrastructure

struct MockCost end

abstract type MockNetworkNodeType end
struct MockNetworkGenerator <: MockNetworkNodeType end
struct MockNetworkDemand <: MockNetworkNodeType end
struct MockNetworkNode <: IS.InfrastructureSystemsComponent
    type::MockNetworkNodeType
    id::Integer
    active_power_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}}
    current_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}}
end
MockNetworkNode(id, pmax) = 
    MockNetworkNode(
        MockNetworkGenerator(),
        id,
        (min = 0.0, max = pmax),
        (min = 0.0, max = 1.0),
    )
MockNetworkNode(id) =
    MockNetworkNode(
        MockNetworkDemand(),
        id,
        (min = 0.0, max = 0.0),
        (min = -1.0, max = 0.0),
    )
IOM.get_name(n::MockNetworkNode) = string(n.id)
IOM.get_base_power(::MockNetworkNode) = 1.0

struct MockPowerModel <: IS.Optimization.AbstractPowerModel end

# ----------------- ContainerTypes

struct KCLConstraint <: ConstraintType end
struct VoltageVariable <: VariableType end
struct CurrentVariable <: VariableType end
IOM.get_variable_binary(
    ::Union{ActivePowerVariable, VoltageVariable, CurrentVariable},
    ::Type{MockNetworkNode},
    ::TestDeviceFormulation,
) = false
IOM.get_variable_lower_bound(
    ::Type{ActivePowerVariable},
    n::MockNetworkNode,
    ::TestDeviceFormulation,
) = n.active_power_limits.min
IOM.get_variable_upper_bound(
    ::Type{ActivePowerVariable},
    g::MockThermalGen,
    ::TestDeviceFormulation,
) = n.active_power_limits.max
IOM.get_variable_lower_bound(
    ::Type{VoltageVariable},
    ::Type{MockNetworkNode},
    ::TestDeviceFormulation,
) = 0.8
IOM.get_variable_upper_bound(
    ::Type{VoltageVariable},
    ::Type{MockNetworkNode},
    ::TestDeviceFormulation,
) = 1.2
IOM.get_variable_lower_bound(
    ::Type{CurrentVariable},
    n::Type{MockNetworkNode},
    ::TestDeviceFormulation,
) = n.current_limits.min
IOM.get_variable_upper_bound(
    ::Type{CurrentVariable},
    n::Type{MockNetworkNode},
    ::TestDeviceFormulation,
) = n.current_limits.max

# ----------------- IOM interface

function IOM.intermediate_set_units_base_system!(::MockSystem, base) end
function IOM.intermediate_get_forecast_initial_timestamp(::MockSystem)
    return DateTime(1970)
end
function IOM.get_available_components(
    ::NetworkModel{MockPowerModel},
    ::Type{T},
    ::MockSystem,
) where {T <: IS.InfrastructureSystemsComponent}
    return 0
end
function IOM.calculate_aux_variables!(::OptimizationContainer, ::MockSystem)
    return IOM.RunStatus.SUCCESSFULLY_FINALIZED
end
function IOM.calculate_dual_variables!(::OptimizationContainer, ::MockSystem, ::Bool)
    return IOM.RunStatus.SUCCESSFULLY_FINALIZED
end

# ----------------- Problem generation

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
    for i in 1:size - 1
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
    for i=1:half
        points = _random_convex_pwl_points(n_cost_segments, pmaxes[i], rng)
        pwl = IS.PiecewiseLinearData(points)
        cost_curve = IS.CostCurve(IS.InputOutputCurve(pwl))
        push!(cost_functions, cost_curve)
    end

    return NetworkProblem(
        size,
        1:half,
        half+1:size,
        edges,
        rand(rng, half),
        conductances,
        cost_functions
    )
end

# ----------------- Container solving

function build_system!(container, sys, problem)
    nodes = Vector{MockNetworkNode}()
    n_gen = div(problem.size, 2)
    for i in 1:n_gen
        node = MockNetworkNode(i, problem.pmaxes[i])
        add_component!(sys, node)
        push!(nodes, node)
    end
    for i in (n_gen + 1):(problem.size)
        node = MockNetworkNode(i)
        add_component!(sys, node)
        push!(nodes, node)
    end
    vars = [ActivePowerVariable, VoltageVariable, CurrentVariable]
    for var in vars
        add_variables!(container, var, nodes, TestDeviceFormulation())
    end
    return nodes
end

function add_problem_constraints!(container, nodes, problem)
    voltages = IOM.get_variable(container, VoltageVariable(), MockNetworkNode)
    currents = IOM.get_variable(container, CurrentVariable(), MockNetworkNode)

    kcl_container = add_constraints_container!(
        container,
        KCLConstraint(),
        MockNetworkNode,
        [string(i) for i=1:problem.size],
        1:2,
        1:1, # time steps
    )

    jump_model = get_jump_model(container)
    for (i, j) in problem.edges
        e = problem.conductances[i, j]
        i, j = string(i), string(j)
        Ii, Ij = currents[i, 1], currents[j, 1]
        Vi, Vj = voltages[i, 1], voltages[j, 1]
        kcl_container[i, 1, 1] = JuMP.@constraint(
            jump_model,
            Ii == e * (Vi - Vj)
        )
        kcl_container[i, 2, 1] = JuMP.@constraint(
            jump_model,
            Ij == e * (Vj - Vi)
        )
    end

    for i in problem.gens
        IOM.add_variable_cost_to_objective!(
            container,
            ActivePowerVariable(),
            nodes[i],
            problem.cost_functions[i],
            TestDeviceFormulation(),
        )
    end
    return
end

function make_container(problem; optimizer = HiGHS.Optimizer)
    sys = MockSystem(1.0)
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

    nodes = build_system!(container, sys, problem)
    add_problem_constraints!(container, nodes, problem)
    return container, sys
end

function generate_network()
    network = NetworkModel(MockPowerModel)
    init_optimization_container!(container, network, sys)
    status = IOM.execute_optimizer!(container, sys)

    return container, status
end

# ----------------- Profiling and comparison

# -----------------

problem = NetworkProblem(; size = 10, n_cost_segments = 3, seed = 0)
container, sys = make_container(problem)
network = NetworkModel(MockPowerModel)
init_optimization_container!(container, network, sys)
IOM.execute_optimizer!(container, sys)