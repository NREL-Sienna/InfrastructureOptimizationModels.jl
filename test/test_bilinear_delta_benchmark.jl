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

struct BenchmarkSystem <: IS.InfrastructureSystemsContainer end
IOM.stores_time_series_in_memory(::BenchmarkSystem) = false
IOM.get_base_power(::BenchmarkSystem) = 1.0

struct VoltageVariable <: VariableType end
struct CurrentVariable <: VariableType end
IOM.get_variable_binary(
    ::Union{ActivePowerVariable, VoltageVariable, CurrentVariable},
    ::Union{Type{MockThermalGen}, Type{MockLoad}},
    ::TestDeviceFormulation,
) = false
IOM.get_variable_lower_bound(
    ::Type{ActivePowerVariable},
    g::MockThermalGen,
    ::TestDeviceFormulation,
) = g.active_power_limits.min
IOM.get_variable_upper_bound(
    ::Type{ActivePowerVariable},
    g::MockThermalGen,
    ::TestDeviceFormulation,
) = g.active_power_limits.max
IOM.get_variable_lower_bound(
    ::Type{VoltageVariable},
    c::Union{Type{MockThermalGen}, Type{MockLoad}},
    ::TestDeviceFormulation,
) = 0.8
IOM.get_variable_upper_bound(
    ::Type{VoltageVariable},
    c::Union{Type{MockThermalGen}, Type{MockLoad}},
    ::TestDeviceFormulation,
) = 1.2
IOM.get_variable_lower_bound(
    ::Type{CurrentVariable},
    ::Type{MockThermalGen},
    ::TestDeviceFormulation,
) = 0.0
IOM.get_variable_upper_bound(
    ::Type{CurrentVariable},
    ::Type{MockThermalGen},
    ::TestDeviceFormulation,
) = 1.0
IOM.get_variable_lower_bound(
    ::Type{CurrentVariable},
    ::Type{MockLoad},
    ::TestDeviceFormulation,
) = -1.0
IOM.get_variable_upper_bound(
    ::Type{CurrentVariable},
    ::Type{MockLoad},
    ::TestDeviceFormulation,
) = 0.0

struct KirchoffsCurrentLawConstraint <: ConstraintType end
struct PowerVoltageCurrentLinkingConstraint <: ConstraintType end
struct MockPowerModel <: IS.Optimization.AbstractPowerModel end
function IOM.intermediate_set_units_base_system!(::MockSystem, base) end
function IOM.intermediate_get_forecast_initial_timestamp(::MockSystem)
    return DateTime(1970)
end
function IOM.get_available_components(
    ::NetworkModel{MockPowerModel},
    ::Type{T},
    ::MockSystem
) where {T <: IS.InfrastructureSystemsComponent}
    return 0
end
function IOM.calculate_aux_variables!(::OptimizationContainer, ::MockSystem)
    return IOM.RunStatus.SUCCESSFULLY_FINALIZED
end
function IOM.calculate_dual_variables!(::OptimizationContainer, ::MockSystem, ::Bool)
    return IOM.RunStatus.SUCCESSFULLY_FINALIZED
end


function generate_network(;
    N::Int = 10,
    K::Int = 3,
    method::Symbol = :sos2,
    refinement::Int = 4,
    seed::Int = 42,
)
    @assert iseven(N) "N must be even"
    rng = MersenneTwister(seed)

    # Initialize system and container
    sys = MockSystem(1.0)
    settings = IOM.Settings(sys; horizon = Dates.Hour(1), resolution = Dates.Hour(1), optimizer = HiGHS.Optimizer)
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, 1:1)

    # Add mock components to system and add active power, voltage, and current
    # variables.
    gen_nodes = Vector{MockThermalGen}()
    dem_nodes = Vector{MockLoad}()
    all_names = Vector{String}()
    for i in 1:(N ÷ 2)
        n_segments = rand(rng, 2:K)
        pmax = 1.5 * rand(rng)
        points = _random_convex_pwl_points(n_segments, pmax, rng)
        pwl = IS.PiecewiseLinearData(points)
        cost_curve = IS.CostCurve(IS.InputOutputCurve(pwl))
        op_cost = MockOperationalCost(cost_curve, 0.0, 0.0)
        name = "gen$i"
        gen = MockThermalGen(
            name, true, (min = 0.0, max = pmax), 1.0, op_cost,
        )
        add_component!(sys, gen)
        push!(gen_nodes, gen)
        push!(all_names, name)
    end
    for i in 1:(N ÷ 2)
        name = "dem$i"
        dem = MockLoad(name, true, 0.5)
        add_component!(sys, dem)
        push!(dem_nodes, dem)
        push!(all_names, name)
    end
    vars = [ActivePowerVariable, VoltageVariable, CurrentVariable]
    nodes = [gen_nodes, dem_nodes]
    for (v, n) in Iterators.product(vars, nodes)
        add_variables!(container, v, n, TestDeviceFormulation())
    end

    # Get variables we just created
    gen_voltages = IOM.get_variable(container, VoltageVariable(), MockThermalGen)
    dem_voltages = IOM.get_variable(container, VoltageVariable(), MockLoad)
    get_voltage =
        n ->
            n in axes(gen_voltages, 1) ? gen_voltages[n, 1] : dem_voltages[n, 1]
    gen_currents = IOM.get_variable(container, CurrentVariable(), MockThermalGen)
    dem_currents = IOM.get_variable(container, CurrentVariable(), MockLoad)
    edges = Dict{String, Vector{String}}()

    # Randomize connectivity
    perm = shuffle(rng, 1:N)
    for i in 1:(N - 1)
        edges[all_names[perm[i]]] = [all_names[perm[i + 1]]]
        edges[all_names[perm[i + 1]]] = [all_names[perm[i]]]
    end
    for _ in 1:(N ÷ 3)
        i, j = shuffle(1:N)[1:2]
        push!(edges[all_names[i]], all_names[j])
        push!(edges[all_names[j]], all_names[i])
    end

    # Build KCL constraints
    gen_kcl_container = add_constraints_container!(
        container,
        KirchoffsCurrentLawConstraint(),
        MockThermalGen,
        ["gen$i" for i in 1:(N ÷ 2)],
        1:1,
    )
    dem_kcl_container = add_constraints_container!(
        container,
        KirchoffsCurrentLawConstraint(),
        MockLoad,
        ["dem$i" for i in 1:(N ÷ 2)],
        1:1,
    )
    jump_model = get_jump_model(container)
    for i in 1:(N ÷ 2)
        name = "gen$i"
        Ii, Vi = gen_currents[name, 1], get_voltage(name)
        for other in edges[name]
            conductance = 0.005 * rand(rng) + 0.005
            Vj = get_voltage(other)

            gen_kcl_container[name, 1] = JuMP.@constraint(
                jump_model,
                Ii == conductance * (Vi - Vj)
            )
        end
    end
    for i in 1:(N ÷ 2)
        name = "dem$i"
        Ii, Vi = dem_currents[name, 1], get_voltage(name)
        for other in edges[name]
            conductance = 0.005 * rand(rng) + 0.005
            Vj = get_voltage(other)

            dem_kcl_container[name, 1] = JuMP.@constraint(
                jump_model,
                Ii == conductance * (Vi - Vj)
            )
        end
    end

    # Add cost
    for g in gen_nodes
        IOM.add_variable_cost_to_objective!(
            container,
            ActivePowerVariable(),
            g,
            get_variable(IOM.get_operation_cost(g)),
            TestDeviceFormulation(),
        )
    end

    # Approximate bilinear terms
    bilinear_fn! = if method === :sos2
        (device, cont, names, xc, yc, ylo, yhi, meta) ->
            IOM._add_sos2_bilinear_approx!(
                cont, device, names, 1:1, xc, yc,
                0.8, 1.2, ylo, yhi, refinement, meta,
            )
    elseif method === :manual_sos2
        (device, cont, names, xc, yc, ylo, yhi, meta) ->
            IOM._add_manual_sos2_bilinear_approx!(
                cont, device, names, 1:1, xc, yc,
                0.8, 1.2, ylo, yhi, refinement, meta,
            )
    elseif method === :sawtooth
        (device, cont, names, xc, yc, ylo, yhi, meta) ->
            IOM._add_sawtooth_bilinear_approx!(
                cont, device, names, 1:1, xc, yc,
                0.8, 1.2, ylo, yhi, refinement, meta,
            )
    else
        error("Unknown method: $method. Use :sos2, :manual_sos2, or :sawtooth.")
    end
    z_gen = bilinear_fn!(
        MockThermalGen,
        container, ["gen$i" for i in 1:(N ÷ 2)], gen_voltages, gen_currents,
        0.0, 1.0, "zgen",
    )
    z_dem = bilinear_fn!(
        MockLoad,
        container, ["dem$i" for i in 1:(N ÷ 2)], dem_voltages, dem_currents,
        -1.0, 0.0, "zdem",
    )

    # Connect bilinear approximations to power
    gen_link_container = add_constraints_container!(
        container,
        PowerVoltageCurrentLinkingConstraint(),
        MockThermalGen,
        ["gen$i" for i in 1:(N ÷ 2)],
        1:1,
    )
    dem_link_container = add_constraints_container!(
        container,
        PowerVoltageCurrentLinkingConstraint(),
        MockLoad,
        ["dem$i" for i in 1:N÷2],
        1:1
    )
    gen_powers = IOM.get_variable(container, ActivePowerVariable(), MockThermalGen)
    dem_powers = IOM.get_variable(container, ActivePowerVariable(), MockLoad)
    for i=1:N÷2
        name = "gen$i"
        gen_link_container[name, 1] = JuMP.@constraint(jump_model, gen_powers[name, 1] == z_gen[(name, 1)])
    end
    for i=1:N÷2
        name = "dem$i"
        dem_link_container[name, 1] = JuMP.@constraint(jump_model, dem_powers[name, 1] == z_dem[(name, 1)])
    end

    network = NetworkModel(MockPowerModel)
    init_optimization_container!(container, network, sys)
    status = IOM.execute_optimizer!(container, sys)

    return container, status
end

container, status = generate_network()
@show objective_value(get_jump_model(container)) status