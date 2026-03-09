# running with julia --project=test [-i] test/test_bilinear_delta_benchmark.jl

using InfrastructureSystems
using InfrastructureOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization
const IOM = InfrastructureOptimizationModels

using HiGHS

# Test directory path for includes
const TEST_DIR = @__DIR__

# Load mock infrastructure (lightweight, no PowerSystems dependency)
include(joinpath(TEST_DIR, "mocks/mock_optimizer.jl"))
include(joinpath(TEST_DIR, "mocks/mock_system.jl"))
include(joinpath(TEST_DIR, "mocks/mock_components.jl"))
include(joinpath(TEST_DIR, "mocks/mock_time_series.jl"))
include(joinpath(TEST_DIR, "mocks/mock_services.jl"))
include(joinpath(TEST_DIR, "mocks/mock_container.jl"))
include(joinpath(TEST_DIR, "mocks/constructors.jl"))
include(joinpath(TEST_DIR, "test_utils/test_types.jl"))
include(joinpath(TEST_DIR, "test_utils/objective_function_helpers.jl"))

struct VoltageVariable <: IOM.VariableType end
struct CurrentVariable <: IOM.VariableType end
struct BilinearProductConstraint <: IOM.ConstraintType end

time_steps = 1:1
sys = make_mock_test_network(10)
println(length(get_components(MockThermalGen, sys)))
settings = IOM.Settings(
    sys;
    horizon = Dates.Hour(length(time_steps)),
    resolution = Dates.Hour(1),
    optimizer = HiGHS.Optimizer,
)
container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
buses = Vector{MockBus}(get_components(MockBus, sys))
generators = Vector{MockThermalGen}(get_components(MockThermalGen, sys))
loads = Vector{MockLoad}(get_components(MockLoad, sys))

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
    ::TestDeviceFormulation
) = c.voltage_limits.min
IOM.get_variable_upper_bound(
    ::Type{VoltageVariable},
    c::Union{Type{MockThermalGen}, Type{MockLoad}},
    ::TestDeviceFormulation
) = c.voltage_limits.max

IOM.get_variable_lower_bound(
    ::Type{CurrentVariable},
    ::Type{MockThermalGen},
    ::TestDeviceFormulation
) = 0.8
IOM.get_variable_upper_bound(
    ::Type{CurrentVariable},
    ::Type{MockThermalGen},
    ::TestDeviceFormulation
) = 1.2
IOM.get_variable_lower_bound(
    ::Type{CurrentVariable},
    ::Type{MockLoad},
    ::TestDeviceFormulation
) = -1.0
IOM.get_variable_upper_bound(
    ::Type{CurrentVariable},
    ::Type{MockLoad},
    ::TestDeviceFormulation
) = 0.0

IOM.get_variable_binary(
    ::Union{ActivePowerVariable, VoltageVariable, CurrentVariable},
    ::Union{Type{MockThermalGen}, Type{MockLoad}},
    ::TestDeviceFormulation,
) = false

add_variables!(
    container,
    ActivePowerVariable,
    generators,
    TestDeviceFormulation(),
)
add_variables!(
    container,
    VoltageVariable,
    generators,
    TestDeviceFormulation()
)
add_variables!(
    container,
    VoltageVariable,
    loads,
    TestDeviceFormulation()
)
add_variables!(
    container,
    CurrentVariable,
    generators,
    TestDeviceFormulation()
)
add_variables!(
    container,
    CurrentVariable,
    loads,
    TestDeviceFormulation()
)

# display(IOM.get_variable(container, VoltageVariable(), MockLoad))

model = get_jump_model(container)
branches = get_components(MockBranch, sys)

function getvar(name, var)
    try
        return IOM.get_variable(container, var, MockLoad)[name, 1]
    catch
    end
    return IOM.get_variable(container, var, MockThermalGen)[name, 1]
end

for node in [generators; loads]
    bus = get_bus(node)
    for branch in branches
        r = get_r(branch)
        from_bus, to_bus = get_from_bus(branch), get_to_bus(branch)
        other_bus = bus == from_bus ? to_bus : (bus == to_bus ? from_bus : nothing)
        if isnothing(other_bus)
            continue
        end
        other_node = nothing
        for n in [generators; loads]
            if get_number(get_bus(n)) == get_number(other_bus)
                other_node = n
            end
        end
        @show node other_node
        current = getvar(get_name(node), CurrentVariable())
        voltage = getvar(get_name(node), VoltageVariable())
        other_voltage = getvar(get_name(other_node), VoltageVariable())
        @constraint(model, current == r * (voltage - other_voltage))
    end
end

# TODO implement true bilinear constraints. Currently we only have quadratic and a few
# types of PWL-approximation-to-bilinear. Then create P_i = V_i * I_i (generators)
# and -d_i = V_i * I_i (loads) constraints.

# TODO add constraints for buses: I_i = sum of 1/r_ij * (V_i - V_j) for all j connected to i.
# model = get_jump_model(container)
# @constraint(model)

for g in generators
    IOM.add_variable_cost_to_objective!(
        container,
        ActivePowerVariable(),
        g,
        get_variable(IOM.get_operation_cost(g)),
        TestDeviceFormulation(),
    )
end

obj = get_objective_expression(container)
println(IOM.get_invariant_terms(obj))

# print(get_jump_model(container))

# TODO check what are the expected units on the cost function.

#=
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

sys = MockSystem(100.0)
bus = MockBus("bus1", 1, :bus1)
IOM.set_time_steps!(container, time_steps)

network = NetworkModel(MockPowerModel)

init_optimization_container!(container, network, sys)
IOM.execute_optimizer!(container, sys)
=#
