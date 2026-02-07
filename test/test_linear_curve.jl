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
    @testset "add_cost_term_invariant! adds to invariant objective" begin
        time_steps = 1:3
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container = setup_container_with_variables(time_steps, device)

        # Add proportional term for time period 1
        rate = 10.0
        name = IOM.get_name(device)
        variable =
            IOM.get_variable(container, TestActivePowerVariable(), MockThermalGen)[name, 1]
        IOM.add_cost_term_invariant!(
            container,
            variable,
            rate,
            TestCostExpression,
            MockThermalGen,
            name,
            1,
        )

        # Verify the coefficient was added to invariant terms
        coef = get_objective_coefficient(
            container,
            TestActivePowerVariable(),
            MockThermalGen,
            "gen1",
            1,
        )
        @test coef ≈ rate

        # Other time periods should have zero coefficient
        for t in 2:3
            coef_t = get_objective_coefficient(
                container,
                TestActivePowerVariable(),
                MockThermalGen,
                "gen1",
                t,
            )
            @test coef_t ≈ 0.0
        end
    end

    @testset "_add_linearcurve_variable_cost! scalar - same cost all time steps" begin
        time_steps = 1:4
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container =
            setup_container_with_variables(time_steps, device; resolution = Dates.Hour(1))

        # Scalar cost: 25.0 $/MWh
        proportional_term = 25.0
        InfrastructureOptimizationModels._add_linearcurve_variable_cost!(
            container,
            TestActivePowerVariable(),
            device,
            proportional_term,
        )

        # With 1-hour resolution, dt = 1.0, so coefficient = proportional_term * dt = 25.0
        expected_coef = proportional_term * 1.0
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable(),
            MockThermalGen,
            "gen1",
            expected_coef,
        )
    end

    @testset "_add_linearcurve_variable_cost! vector - different costs per time step" begin
        time_steps = 1:4
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container =
            setup_container_with_variables(time_steps, device; resolution = Dates.Hour(1))

        # Time-varying costs
        proportional_terms = [10.0, 20.0, 30.0, 40.0]
        InfrastructureOptimizationModels._add_linearcurve_variable_cost!(
            container,
            TestActivePowerVariable(),
            device,
            proportional_terms,
        )

        # With 1-hour resolution, dt = 1.0
        expected_coefs = proportional_terms .* 1.0
        @test verify_objective_coefficients(
            container,
            TestActivePowerVariable(),
            MockThermalGen,
            "gen1",
            expected_coefs,
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
