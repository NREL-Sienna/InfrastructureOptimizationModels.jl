"""
Unit tests for linear curve objective function construction.
Tests the functions in src/objective_function/linear_curve.jl using mock components.
"""

# Test-specific variable type
struct TestActivePowerVariable <: InfrastructureOptimizationModels.VariableType end

# Test-specific formulation
struct TestLinearFormulation <: InfrastructureOptimizationModels.AbstractDeviceFormulation end

# Stub: objective_function_multiplier returns 1.0 for test types
InfrastructureOptimizationModels.objective_function_multiplier(
    ::TestActivePowerVariable,
    ::TestLinearFormulation,
) = 1.0

# Helper to set up container with variables for a device
function setup_container_with_variables(
    time_steps::UnitRange{Int},
    device::MockThermalGen;
    resolution = Dates.Hour(1),
)
    sys = MockSystem(100.0)
    settings = InfrastructureOptimizationModels.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = resolution,
    )
    container = InfrastructureOptimizationModels.OptimizationContainer(
        sys,
        settings,
        JuMP.Model(),
        MockDeterministic,
    )
    InfrastructureOptimizationModels.set_time_steps!(container, time_steps)

    # Add variable container for the device
    device_name = get_name(device)
    var_container = InfrastructureOptimizationModels.add_variable_container!(
        container,
        TestActivePowerVariable(),
        MockThermalGen,
        [device_name],
        time_steps,
    )

    # Populate with actual JuMP variables
    jump_model = InfrastructureOptimizationModels.get_jump_model(container)
    for t in time_steps
        var_container[device_name, t] = JuMP.@variable(
            jump_model,
            base_name = "TestActivePower_$(device_name)_$(t)",
        )
    end

    return container
end

@testset "Linear Curve Objective Functions" begin
    @testset "add_proportional_cost_invariant! with NATURAL_UNITS" begin
        time_steps = 1:4
        device = make_mock_thermal("gen1"; base_power = 50.0)
        container =
            setup_container_with_variables(time_steps, device; resolution = Dates.Hour(1))

        # Cost: 25.0 $/MWh in natural units, system base = 100 MW
        # Normalized: 25.0 * 100.0 = 2500.0 $/p.u.h
        # With dt = 1.0 and multiplier = 1.0: coefficient = 2500.0
        InfrastructureOptimizationModels.add_proportional_cost_invariant!(
            container,
            TestActivePowerVariable,
            device,
            25.0,
            IS.UnitSystem.NATURAL_UNITS,
        )

        expected_coef = 25.0 * 100.0 * 1.0
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable(),
            MockThermalGen,
            "gen1",
            expected_coef,
        )
    end

    @testset "add_proportional_cost_invariant! with multiplier and sub-hourly resolution" begin
        time_steps = 1:4
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container =
            setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Minute(15),
            )

        # Cost: 20.0 $/MWh in system base, multiplier = 2.0
        # dt = 0.25, so coefficient = 20.0 * 2.0 * 0.25 = 10.0
        InfrastructureOptimizationModels.add_proportional_cost_invariant!(
            container,
            TestActivePowerVariable,
            device,
            20.0,
            IS.UnitSystem.SYSTEM_BASE,
            2.0,
        )

        expected_coef = 20.0 * 2.0 * 0.25
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable(),
            MockThermalGen,
            "gen1",
            expected_coef,
        )
    end

    @testset "add_proportional_cost_invariant! skips zero cost" begin
        time_steps = 1:3
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container =
            setup_container_with_variables(time_steps, device; resolution = Dates.Hour(1))

        InfrastructureOptimizationModels.add_proportional_cost_invariant!(
            container,
            TestActivePowerVariable,
            device,
            0.0,
            IS.UnitSystem.NATURAL_UNITS,
        )

        # Should have zero coefficients since cost_term is zero
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable(),
            MockThermalGen,
            "gen1",
            0.0,
        )
    end

    @testset "add_variable_cost_to_objective! with CostCurve{LinearCurve}" begin
        @testset "NATURAL_UNITS" begin
            time_steps = 1:3
            # Device with 50 MW base power, system has 100 MW base
            device = make_mock_thermal("gen1"; base_power = 50.0)
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Hour(1),
            )

            # Cost: 30 $/MWh in natural units (MW)
            cost_curve = IS.CostCurve(
                IS.LinearCurve(30.0),
                IS.UnitSystem.NATURAL_UNITS,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable(),
                device,
                cost_curve,
                TestLinearFormulation(),
            )

            # NATURAL_UNITS: cost is in $/MW, variable is in p.u. (system base)
            # proportional_term_per_unit = 30.0 * 100.0 (system_base) = 3000.0
            # With dt = 1.0, coefficient = 3000.0
            expected_coef = 30.0 * 100.0 * 1.0
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable(),
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end

        @testset "SYSTEM_BASE" begin
            time_steps = 1:3
            device = make_mock_thermal("gen1"; base_power = 50.0)
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Hour(1),
            )

            # Cost: 30 $/p.u.h in system base units
            cost_curve = IS.CostCurve(
                IS.LinearCurve(30.0),
                IS.UnitSystem.SYSTEM_BASE,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable(),
                device,
                cost_curve,
                TestLinearFormulation(),
            )

            # SYSTEM_BASE: cost is already in $/p.u., no conversion needed
            # proportional_term_per_unit = 30.0
            # With dt = 1.0, coefficient = 30.0
            expected_coef = 30.0 * 1.0
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable(),
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end

        @testset "DEVICE_BASE" begin
            time_steps = 1:3
            # Device with 50 MW base power, system has 100 MW base
            device = make_mock_thermal("gen1"; base_power = 50.0)
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Hour(1),
            )

            # Cost: 30 $/p.u.h in device base units
            cost_curve = IS.CostCurve(
                IS.LinearCurve(30.0),
                IS.UnitSystem.DEVICE_BASE,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable(),
                device,
                cost_curve,
                TestLinearFormulation(),
            )

            # DEVICE_BASE: cost is in $/device_p.u., variable is in system p.u.
            # To convert: cost * (system_base / device_base)
            # proportional_term_per_unit = 30.0 * (100/50) = 60.0
            # With dt = 1.0, coefficient = 60.0
            expected_coef = 30.0 * (100.0 / 50.0) * 1.0
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable(),
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end

        @testset "with non-unity resolution" begin
            time_steps = 1:3
            device = make_mock_thermal("gen1"; base_power = 100.0)
            # 15-minute resolution
            container = setup_container_with_variables(
                time_steps,
                device;
                resolution = Dates.Minute(15),
            )

            # Cost: 20 $/MWh in natural units
            cost_curve = IS.CostCurve(
                IS.LinearCurve(20.0),
                IS.UnitSystem.NATURAL_UNITS,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestActivePowerVariable(),
                device,
                cost_curve,
                TestLinearFormulation(),
            )

            # NATURAL_UNITS with 15-min resolution:
            # proportional_term_per_unit = 20.0 * 100.0 = 2000.0
            # dt = 15 minutes / 60 = 0.25 hours
            # coefficient = 2000.0 * 0.25 = 500.0
            expected_coef = 20.0 * 100.0 * 0.25
            @test verify_objective_coefficients(
                container,
                TestActivePowerVariable(),
                MockThermalGen,
                "gen1",
                expected_coef,
            )
        end
    end

    @testset "add_variable_cost_to_objective! with FuelCurve{LinearCurve}" begin
        time_steps = 1:3
        device = make_mock_thermal("gen1"; base_power = 50.0)
        container = setup_container_with_variables(
            time_steps,
            device;
            resolution = Dates.Hour(1),
        )

        # FuelCurve: fuel consumption rate (MMBTU/MWh) × fuel cost ($/MMBTU)
        # Linear fuel consumption: 8.0 MMBTU/MWh
        # Fuel cost: 5.0 $/MMBTU
        # Total cost: 8.0 * 5.0 = 40.0 $/MWh
        fuel_curve = IS.FuelCurve(
            IS.LinearCurve(8.0),  # MMBTU/MWh
            IS.UnitSystem.NATURAL_UNITS,
            5.0,  # $/MMBTU
        )

        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestActivePowerVariable(),
            device,
            fuel_curve,
            TestLinearFormulation(),
        )

        # NATURAL_UNITS: fuel_curve_per_unit = 8.0 * 100.0 (system_base) = 800.0
        # Total cost coefficient = fuel_curve_per_unit * fuel_cost * dt
        #                        = 800.0 * 5.0 * 1.0 = 4000.0
        expected_coef = 8.0 * 100.0 * 5.0 * 1.0
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable(),
            MockThermalGen,
            "gen1",
            expected_coef,
        )
    end
end
