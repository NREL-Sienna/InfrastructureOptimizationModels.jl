using Test
using PowerOptimizationModels
using Logging

# Import InfrastructureSystems for logging utilities
using InfrastructureSystems
const IS = InfrastructureSystems

# Code Quality Tests (optional, don't block main tests)
import Aqua
@testset "Code Quality (Aqua.jl)" begin
    Aqua.test_ambiguities(PowerOptimizationModels)
    Aqua.find_persistent_tasks_deps(PowerOptimizationModels)
    Aqua.test_persistent_tasks(PowerOptimizationModels)
    Aqua.test_unbound_args(PowerOptimizationModels)
    # Note: These tests are known to fail and are tracked separately:
    # - test_undefined_exports: Has 216 undefined exports (PowerSimulations types)
    # - test_stale_deps: Has 2 stale dependencies (Distributions, ProgressMeter)
end

# Load the test module
include("PowerOptimizationModelsTests.jl")

# Run the test suite
logger = global_logger()

try
    PowerOptimizationModelsTests.run_tests()
finally
    # Guarantee that the global logger is reset
    global_logger(logger)
    nothing
end
