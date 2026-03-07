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
generators = Vector{MockThermalGen}(get_components(MockThermalGen, sys))
loads = get_components(MockLoad, sys)

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
IOM.get_variable_binary(
    ::ActivePowerVariable,
    ::Type{MockThermalGen},
    ::TestDeviceFormulation,
) = false
# TODO: voltage, current variables at each bus
# (do those already exist in IOM? if not make mocks)

add_variables!(
    container,
    ActivePowerVariable,
    generators,
    TestDeviceFormulation(),
)

# TODO implement true bilinear constraints. Currently we only have quadratic and a few
# types of PWL-approximation-to-bilinear. Then create P_i = V_i * I_i (generators)
# and -d_i = V_i * I_i (loads) constraints.

# TODO add constraints for buses: I_i = sum of 1/r_ij * (V_i - V_j) for all j connected to i.

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
