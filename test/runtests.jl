using Test
using InfrastructureOptimizationModels
using Logging

# Import InfrastructureSystems for logging utilities
using InfrastructureSystems
const IS = InfrastructureSystems

# Code Quality Tests (optional, don't block main tests)
import Aqua
@testset "Code Quality (Aqua.jl)" begin
    Aqua.test_ambiguities(InfrastructureOptimizationModels)
    Aqua.find_persistent_tasks_deps(InfrastructureOptimizationModels)
    Aqua.test_persistent_tasks(InfrastructureOptimizationModels)
    Aqua.test_unbound_args(InfrastructureOptimizationModels)
    Aqua.test_exported_names(InfrastructureOptimizationModels)
    Aqua.test_stale_deps(InfrastructureOptimizationModels)
end

# Load the test module
include("InfrastructureOptimizationModelsTests.jl")

# Run the test suite
logger = global_logger()

try
    InfrastructureOptimizationModelsTests.run_tests()
finally
    # Guarantee that the global logger is reset
    global_logger(logger)
    nothing
end
