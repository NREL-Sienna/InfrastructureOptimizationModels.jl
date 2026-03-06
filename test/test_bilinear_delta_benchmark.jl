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

time_steps = 1:1
sys = MockSystem(100.0)
bus = MockBus("bus1", 1, :bus1)
settings = IOM.Settings(
    sys;
    horizon = Dates.Hour(length(time_steps)),
    resolution = Dates.Hour(1),
    optimizer = HiGHS.Optimizer
)
container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
IOM.set_time_steps!(container, time_steps)

network = NetworkModel(MockPowerModel)

init_optimization_container!(container, network, sys)
IOM.execute_optimizer!(container, sys)

"end"