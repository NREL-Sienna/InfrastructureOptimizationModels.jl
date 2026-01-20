module PowerOptimizationModelsTests

#=
Testing Strategy:
- Lightweight tests run first using only mock objects (no PowerSystems types used)
- Then tests that require PowerSystems types run
- All heavy dependencies are loaded at module level (Julia requirement)
  but tests are organized to run lightweight ones first
=#

using Test
using PowerOptimizationModels
using Logging
using Dates

# Import InfrastructureSystems for logging utilities
using InfrastructureSystems
const IS = InfrastructureSystems

# Load mock infrastructure (lightweight, no PowerSystems dependency)
include("mocks/mock_optimizer.jl")
include("mocks/mock_system.jl")
include("mocks/mock_components.jl")
include("mocks/mock_time_series.jl")
include("mocks/mock_services.jl")
include("mocks/constructors.jl")

# Environment flags for test selection
const RUN_UNIT_TESTS = get(ENV, "POM_RUN_UNIT_TESTS", "true") == "true"
const RUN_INTEGRATION_TESTS = get(ENV, "POM_RUN_INTEGRATION_TESTS", "false") == "true"

# Heavy dependencies - only load if we need tests that use them
if RUN_INTEGRATION_TESTS
    using PowerSystems
    using JuMP
    const PSY = PowerSystems
end

const LOG_FILE = "power-optimization-models-test.log"

# Logging configuration
function setup_logging()
    logging_config_filename = get(ENV, "SIIP_LOGGING_CONFIG", nothing)
    if logging_config_filename !== nothing
        config = IS.LoggingConfiguration(logging_config_filename)
    else
        config = IS.LoggingConfiguration(;
            filename = LOG_FILE,
            file_level = Logging.Info,
            console_level = Logging.Error,
        )
    end
    return config
end

# Main test execution
function run_tests()
    config = setup_logging()
    console_logger = ConsoleLogger(config.console_stream, config.console_level)

    IS.open_file_logger(LOG_FILE, config.file_level) do file_logger
        levels = (Logging.Info, Logging.Warn, Logging.Error)
        multi_logger =
            IS.MultiLogger([console_logger, file_logger], IS.LogEventTracker(levels))
        global_logger(multi_logger)

        if !isempty(config.group_levels)
            IS.set_group_levels!(multi_logger, config.group_levels)
        end

        @info "Running PowerOptimizationModels.jl tests"
        @info "Unit tests: $RUN_UNIT_TESTS"
        @info "Integration tests: $RUN_INTEGRATION_TESTS"

        if RUN_UNIT_TESTS
            @info "Starting unit tests..."
            @time @testset "PowerOptimizationModels Unit Tests" begin
                # Lightweight tests (use only mock objects, no PSY types)
                @testset "Lightweight Tests (mocks only)" begin
                    @info "Running lightweight tests..."
                    include("test_settings.jl")
                end

                # Tests requiring PowerSystems types
                if RUN_INTEGRATION_TESTS
                    @testset "Tests with PowerSystems" begin
                        @info "Running tests that require PowerSystems..."
                        include("test_device_model.jl")
                        include("test_optimization_container.jl")
                        include("test_pwl_methods.jl")
                    end
                end
            end
        end

        # Integration tests - placeholder for future
        if RUN_INTEGRATION_TESTS
            @info "Starting integration tests..."
            @info "Note: Integration tests not yet implemented"
        end

        @test length(IS.get_log_events(multi_logger.tracker, Logging.Error)) == 0
        @info IS.report_log_summary(multi_logger)
    end
end

end # module PowerOptimizationModelsTests
