# InfrastructureOptimizationModels.jl

Library for Optimization modeling in Sienna. This is a utility library that defines useful objects and
routines for managing power optimization models. Julia compat: `^1.10`.

> **General Sienna Programming Practices:** For information on performance requirements, code conventions, documentation practices, and contribution workflows that apply across all Sienna packages, see [Sienna.md](Sienna.md). Always
check this file before making plans or changes.

> **Maintenance note:** This file documents the repository structure and conventions. Update it
whenever files, directories, or architectural patterns change so it stays accurate.

## Repository Structure

```
Project.toml              # Package manifest (dependencies, compat)
src/
  InfrastructureOptimizationModels.jl   # Main module file (exports & includes)
  core/                   # Foundational types and containers
    definitions.jl            # Constants, enums, type aliases
    optimization_container.jl # Central JuMP-backed optimization container
    optimization_container_keys.jl
    optimization_container_types.jl
    optimization_container_metadata.jl
    dataset.jl / dataset_container.jl  # Result dataset storage
    device_model.jl           # DeviceModel parametric wrapper
    network_model.jl          # NetworkModel parametric wrapper
    service_model.jl          # ServiceModel parametric wrapper
    settings.jl               # Solver/model settings
    model_internal.jl         # Internal model bookkeeping
    model_store_params.jl     # Store parameter definitions
    abstract_model_store.jl   # Abstract store interface
    initial_conditions.jl     # Initial condition types
    parameter_container.jl    # Parameter container type
    operation_model_abstract_types.jl  # Abstract supertypes for models
    optimization_problem_results.jl    # Results access layer
    optimization_problem_results_export.jl
    optimizer_stats.jl        # Solve statistics tracking
    results_by_time.jl        # Time-indexed results cache
    standard_variables_expressions.jl  # Standard variable/expression names
    time_series_parameter_types.jl     # Time series parameter wrappers
    network_reductions.jl     # Network reduction utilities
  common_models/          # Reusable model-building methods
    add_variable.jl           # Variable creation helpers
    add_auxiliary_variable.jl # Auxiliary variable helpers
    add_jump_expressions.jl   # JuMP expression builders
    add_param_container.jl    # Parameter container builders
    add_constraint_dual.jl    # Constraint & dual helpers
    add_pwl_methods.jl        # Piecewise-linear variable/constraint methods
    constraint_helpers.jl     # Generic constraint utilities
    range_constraint.jl       # Min/max range constraints
    duration_constraints.jl   # Min up/down time constraints
    rateofchange_constraints.jl # Ramp rate constraints
    set_expression.jl         # Expression assignment helpers
    get_time_series.jl        # Time series retrieval
    interfaces.jl             # Abstract interface definitions
  initial_conditions/     # Initial condition logic
    add_initial_condition.jl
    calculate_initial_condition.jl
    initialization.jl
  objective_function/     # Objective function construction
    common.jl                 # Shared objective utilities
    cost_term_helpers.jl      # Cost curve → JuMP term conversion
    import_export.jl          # Import/export cost handling
    linear_curve.jl           # LinearCurve objectives
    quadratic_curve.jl        # QuadraticCurve objectives
    piecewise_linear.jl       # PiecewiseLinearCurve objectives
    proportional.jl           # Proportional cost objectives
    market_bid.jl             # Market bid cost objectives
    offer_curve_types.jl      # Offer curve type handling
    start_up_shut_down.jl     # Start-up/shut-down cost terms
  operation/              # Operation model types and workflows
    decision_model.jl         # DecisionModel (single-period optimization)
    decision_model_store.jl   # DecisionModel result store
    emulation_model.jl        # EmulationModel (rolling horizon)
    emulation_model_store.jl  # EmulationModel result store
    operation_model_interface.jl        # Shared model interface methods
    operation_model_serialization.jl    # Serialization/deserialization
    problem_template.jl       # ProblemTemplate (model specification)
    problem_results.jl        # Result post-processing
    store_common.jl           # Shared store utilities
    time_series_interface.jl  # Time series integration
    initial_conditions_update_in_memory_store.jl
    model_numerical_analysis_utils.jl
    optimization_debugging.jl # Debug/diagnostic tools
  utils/                  # General-purpose utilities
    jump_utils.jl             # JuMP helper functions
    dataframes_utils.jl       # DataFrame manipulation
    datetime_utils.jl         # Date/time helpers
    file_utils.jl             # File I/O utilities
    logging.jl                # Logging setup
    indexing.jl               # Index/key utilities
    powersystems_utils.jl     # PowerSystems integration utilities
    time_series_utils.jl      # Time series helpers
    generate_valid_formulations.jl # Formulation validation
    print_pt_v2.jl / print_pt_v3.jl   # Pretty-printing
test/
  runtests.jl             # Entry point — calls load_tests.jl
  load_tests.jl           # Discovers and includes test files
  InfrastructureOptimizationModelsTests.jl  # Test module (imports, aliases, includes)
  includes.jl             # Includes mocks and test utilities
  verify_mocks.jl         # Validates mock types match real interfaces
  test_*.jl               # Individual test files (one per feature area)
  mocks/                  # Mock types for testing without PowerSystems
    mock_components.jl        # MockThermalGen, MockRenewableGen, etc.
    mock_system.jl            # MockSystem container
    mock_container.jl         # Mock optimization container
    mock_optimizer.jl         # Mock solver
    mock_services.jl          # Mock service components
    mock_time_series.jl       # Mock time series data
    constructors.jl           # Convenience constructors for mocks
  test_utils/             # Shared test helpers and fixtures
    common_operation_model.jl
    mock_operation_models.jl
    operations_problem_templates.jl
    solver_definitions.jl
    objective_function_helpers.jl
    model_checks.jl
    test_types.jl
    add_market_bid_cost.jl
    mbc_system_utils.jl
    iec_simulation_utils.jl
    run_simulation.jl
  performance/            # Performance benchmarks
    performance_test.jl
docs/                     # Documenter.jl documentation
  make.jl                 # Build script
  Project.toml            # Docs dependencies
  src/
    index.md
    tutorials/
    how_to_guides/
    explanation/
    reference/
scripts/formatter/        # Code formatting (JuliaFormatter)
```

### Key architectural notes

- **`src/core/`** defines foundational types (`OptimizationContainer`, `DeviceModel`,
  `NetworkModel`, `ServiceModel`, `Settings`, etc.) that are used throughout the package.
- **`src/common_models/`** provides reusable constraint/variable/expression builders that
  concrete formulations call into.
- **`src/objective_function/`** translates cost curves into JuMP objective terms. Each cost
  curve type has its own file.
- **`src/operation/`** implements `DecisionModel` and `EmulationModel` — the two main model
  types — plus serialization, result stores, and the problem template.
- **`src/utils/`** is for pure utility functions with no domain coupling.
- **`test/mocks/`** provides lightweight stand-ins so tests don't depend on PowerSystems
  concrete types.

## Type and Function Conventions

**Prefer IS types over PSY types:** When possible, use InfrastructureSystems parent types:
- `PSY.Component` → `IS.InfrastructureSystemsComponent`
- `PSY.System` → `IS.InfrastructureSystemsContainer`
- Cost curves: `IS.CostCurve`, `IS.LinearCurve`, `IS.UnitSystem`, etc.

## Testing

**Test file structure:** Test files are included by `InfrastructureOptimizationModelsTests.jl`, which
handles imports and mock infrastructure. Don't add `using`, `include`, or `const` alias statements
at the top of individual test files.

**Use mocks over PSY types:** Tests should use mock components (`MockThermalGen`, `MockSystem`, etc.)
rather than PowerSystems types when possible.
