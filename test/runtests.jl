using Test
using InfrastructureOptimizationModels
using Logging
using CoverageTools
using Coverage

const BASE_DIR = joinpath(@__DIR__, "..")

function is_running_on_ci()
    return get(ENV, "CI", "false") == "true" || haskey(ENV, "GITHUB_ACTIONS")
end

const LOCAL_COVERAGE = Base.JLOptions().code_coverage != 0 && !is_running_on_ci()
if LOCAL_COVERAGE
    CoverageTools.clean_folder(joinpath(BASE_DIR, "src"))
end

# Import InfrastructureSystems for logging utilities
using InfrastructureSystems
const IS = InfrastructureSystems

# Code Quality Tests
import Aqua
@testset "Code Quality (Aqua.jl)" begin
    Aqua.test_all(InfrastructureOptimizationModels; persistent_tasks = false)
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
    if LOCAL_COVERAGE
        coverage = CoverageTools.process_folder(joinpath(BASE_DIR, "src"))
        LCOV.writefile(joinpath(BASE_DIR, "lcov.info"), coverage)
    end
end
