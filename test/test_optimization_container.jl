"""
Unit tests for OptimizationContainer using mock components.
Tests container machinery without requiring real PowerSystems data or solvers.
"""

using InfrastructureSystems

# Define aliases if not already defined by test harness
if !@isdefined(PSI)
    const PSI = InfrastructureOptimizationModels
end
if !@isdefined(PSY)
    using PowerSystems
    const PSY = PowerSystems
end
const ISOPT = InfrastructureSystems.Optimization

# Mock constraint/expression types for testing container machinery
struct MockConstraintType <: ISOPT.ConstraintType end
struct MockExpressionType <: ISOPT.ExpressionType end

@testset "OptimizationContainer with MockSystem" begin
    @testset "Container creation" begin
        # Create mock system
        mock_sys = MockSystem(100.0)

        # Create settings with mock system
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )

        # Create container - uses duck-typed system
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            PSY.Deterministic,
        )

        @test PSI.get_model_base_power(container) == 100.0
        @test isempty(PSI.get_variables(container))
        @test isempty(PSI.get_constraints(container))
        @test isempty(PSI.get_expressions(container))
    end

    @testset "add_variable_container!" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            PSY.Deterministic,
        )

        # Set time steps (normally done by init_optimization_container!)
        PSI.set_time_steps!(container, 1:24)

        # Add a variable container using a real PSY component type as the key
        # (the container just needs a type - doesn't need actual component instances)
        device_names = ["gen1", "gen2", "gen3"]
        time_steps = PSI.get_time_steps(container)

        var_container = PSI.add_variable_container!(
            container,
            PSI.ActivePowerVariable(),
            PSY.ThermalStandard,
            device_names,
            time_steps,
        )

        # Verify the container was created
        @test !isempty(PSI.get_variables(container))

        # Verify we can retrieve it
        var_key = PSI.VariableKey(PSI.ActivePowerVariable, PSY.ThermalStandard)
        @test haskey(PSI.get_variables(container), var_key)

        # Verify dimensions
        retrieved =
            PSI.get_variable(container, PSI.ActivePowerVariable(), PSY.ThermalStandard)
        @test size(retrieved) == (length(device_names), length(time_steps))
    end

    @testset "add_constraints_container!" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            PSY.Deterministic,
        )
        PSI.set_time_steps!(container, 1:24)

        device_names = ["gen1", "gen2"]
        time_steps = PSI.get_time_steps(container)

        cons_container = PSI.add_constraints_container!(
            container,
            MockConstraintType(),
            PSY.ThermalStandard,
            device_names,
            time_steps,
        )

        @test !isempty(PSI.get_constraints(container))

        cons_key = PSI.ConstraintKey(MockConstraintType, PSY.ThermalStandard)
        @test haskey(PSI.get_constraints(container), cons_key)
    end

    @testset "add_expression_container!" begin
        mock_sys = MockSystem(100.0)
        settings = PSI.Settings(
            mock_sys;
            horizon = Dates.Hour(24),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,  # Bypass stores_time_series_in_memory check
        )
        container = PSI.OptimizationContainer(
            mock_sys,
            settings,
            nothing,
            PSY.Deterministic,
        )
        PSI.set_time_steps!(container, 1:24)

        device_names = ["gen1", "gen2"]
        time_steps = PSI.get_time_steps(container)

        expr_container = PSI.add_expression_container!(
            container,
            MockExpressionType(),
            PSY.ThermalStandard,
            device_names,
            time_steps,
        )

        @test !isempty(PSI.get_expressions(container))

        expr_key = PSI.ExpressionKey(MockExpressionType, PSY.ThermalStandard)
        @test haskey(PSI.get_expressions(container), expr_key)
    end
end
