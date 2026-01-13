module PowerOptimizationModelsTests

#=
Testing Strategy:
- Unit tests use mock objects and avoid heavy operations
- Focus on API surface, method signatures, and lightweight operations
- Integration tests (opt-in) use real PowerSystems for full functionality
=#

using Test
using PowerOptimizationModels
using Logging
using Dates

# Import InfrastructureSystems for logging utilities
using InfrastructureSystems
const IS = InfrastructureSystems

# Load mock infrastructure
include("mocks/mock_system.jl")
include("mocks/mock_components.jl")
include("mocks/mock_time_series.jl")
include("mocks/mock_services.jl")
include("mocks/constructors.jl")

# Environment flags for test selection
const RUN_UNIT_TESTS = get(ENV, "POM_RUN_UNIT_TESTS", "true") == "true"
const RUN_INTEGRATION_TESTS = get(ENV, "POM_RUN_INTEGRATION_TESTS", "false") == "true"

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

        # Unit tests (fast, no external dependencies except test mocks)
        if RUN_UNIT_TESTS
            @info "Starting unit tests..."
            @time @testset "PowerOptimizationModels Unit Tests" begin
                # Core infrastructure tests
                @testset "Core" begin
                    test_files = [
                        "unit/core/test_optimization_container.jl",
                        "unit/core/test_device_model.jl",
                        "unit/core/test_network_model.jl",
                        "unit/core/test_problem_template.jl",
                        "unit/core/test_settings.jl",
                    ]
                    for test_file in test_files
                        if isfile(test_file)
                            @info "Running $test_file"
                            include(test_file)
                        end
                    end
                end

                # Results and I/O tests
                @testset "Operation" begin
                    test_files = [
                        # Add as they are created:
                        # "unit/operation/test_problem_results.jl",
                        # "unit/operation/test_results_by_time.jl",
                        # "unit/operation/test_serialization.jl",
                    ]
                    for test_file in test_files
                        if isfile(test_file)
                            @info "Running $test_file"
                            include(test_file)
                        end
                    end
                end
            end
        end

        # Integration tests - for now just placeholder
        if RUN_INTEGRATION_TESTS
            @info "Starting integration tests..."
            @info "Note: Integration tests not yet implemented"
            # Integration tests will be added later when needed
        end

        @test length(IS.get_log_events(multi_logger.tracker, Logging.Error)) == 0
        @info IS.report_log_summary(multi_logger)
    end
end

end # module PowerOptimizationModelsTests
