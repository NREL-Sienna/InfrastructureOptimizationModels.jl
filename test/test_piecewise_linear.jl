"""
Unit tests for piecewise linear objective function construction.
Tests the functions in src/objective_function/piecewise_linear.jl using mock components.
"""

# Test-specific variable type
if !isdefined(InfrastructureOptimizationModelsTests, :TestPWLVariable)
    struct TestPWLVariable <: InfrastructureOptimizationModels.VariableType end
end

# Test-specific formulation
if !isdefined(InfrastructureOptimizationModelsTests, :TestPWLFormulation)
    struct TestPWLFormulation <: InfrastructureOptimizationModels.AbstractDeviceFormulation end
end

# Required stubs
InfrastructureOptimizationModels.objective_function_multiplier(
    ::TestPWLVariable,
    ::TestPWLFormulation,
) = 1.0

InfrastructureOptimizationModels.sos_status(
    ::MockThermalGen,
    ::TestPWLFormulation,
) = IOM.SOSStatusVariable.NO_VARIABLE

# Helper to set up container with variables for a device
function setup_pwl_container_with_variables(
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
        TestPWLVariable(),
        MockThermalGen,
        [device_name],
        time_steps,
    )

    # Populate with actual JuMP variables
    jump_model = InfrastructureOptimizationModels.get_jump_model(container)
    for t in time_steps
        var_container[device_name, t] = JuMP.@variable(
            jump_model,
            base_name = "TestPWLPower_$(device_name)_$(t)",
        )
    end

    return container
end

# Standard PWL points for testing
const CONVEX_PWL_POINTS = [
    (x = 0.0, y = 0.0),
    (x = 0.5, y = 10.0),
    (x = 1.0, y = 25.0),
]  # Slopes: 20, 30 (convex - increasing)

const NONCONVEX_PWL_POINTS = [
    (x = 0.0, y = 0.0),
    (x = 0.5, y = 20.0),
    (x = 1.0, y = 25.0),
]  # Slopes: 40, 10 (non-convex - decreasing)

"""
Helper to set up common test fixtures for PWL tests.
Returns (container, device, cost_curve, pwl_data).
"""
function setup_pwl_test(;
    time_steps = 1:2,
    device_name = "gen1",
    device_base_power = 100.0,
    resolution = Dates.Hour(1),
    points = CONVEX_PWL_POINTS,
    unit_system = IS.UnitSystem.NATURAL_UNITS,
    fuel_cost = nothing,  # If set, creates FuelCurve instead of CostCurve
)
    # When fuel_cost is provided, the device's operation_cost must also have it
    # because get_fuel_cost(device) is called to look up the cost multiplier
    if isnothing(fuel_cost)
        op_cost = MockOperationCost(0.0, false, 0.0)
    else
        op_cost = MockOperationCost(0.0, false, fuel_cost)
    end
    device = make_mock_thermal(
        device_name;
        base_power = device_base_power,
        operation_cost = op_cost,
    )
    container = setup_pwl_container_with_variables(time_steps, device; resolution)
    pwl_data = IS.PiecewiseLinearData(points)

    if isnothing(fuel_cost)
        cost_curve = IS.CostCurve(
            IS.InputOutputCurve(pwl_data),
            unit_system,
        )
    else
        cost_curve = IS.FuelCurve(
            IS.InputOutputCurve(pwl_data),
            unit_system,
            fuel_cost,
        )
    end

    return (; container, device, cost_curve, pwl_data)
end

@testset "Piecewise Linear Objective Functions" begin
    @testset "_add_pwl_variables!" begin
        (; container, pwl_data) = setup_pwl_test()

        # Add PWL variables for time period 1
        pwl_vars = InfrastructureOptimizationModels._add_pwl_variables!(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
        )

        # length(PiecewiseLinearData) = number of segments; code adds +1 to get number of points
        @test length(pwl_vars) == length(pwl_data) + 1

        # Verify bounds are [0, 1]
        for var in pwl_vars
            @test JuMP.lower_bound(var) == 0.0
            @test JuMP.upper_bound(var) == 1.0
        end

        # Verify variables are stored in PiecewiseLinearCostVariable container
        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
            MockThermalGen,
        )
        @test !isnothing(pwl_var_container)
        @test pwl_var_container["gen1", 1, 1] === pwl_vars[1]
    end

    @testset "_add_pwl_constraint_standard! creates linking and normalization constraints" begin
        (; container, device, pwl_data) = setup_pwl_test(; time_steps = 1:1)

        # First add the PWL variables
        InfrastructureOptimizationModels._add_pwl_variables!(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
        )

        # Add the PWL constraint
        break_points = IS.get_x_coords(pwl_data)
        power_var =
            IOM.get_variable(container, TestPWLVariable(), MockThermalGen)["gen1", 1]
        InfrastructureOptimizationModels._add_pwl_constraint_standard!(
            container,
            device,
            break_points,
            IOM.SOSStatusVariable.NO_VARIABLE,
            1,
            power_var,
        )

        # Check that constraints were added
        jump_model = InfrastructureOptimizationModels.get_jump_model(container)
        @test JuMP.num_constraints(jump_model, JuMP.AffExpr, MOI.EqualTo{Float64}) >= 2

        # Verify constraint containers exist and contain constraint refs
        const_container = InfrastructureOptimizationModels.get_constraint(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostConstraint(),
            MockThermalGen,
        )
        @test const_container["gen1", 1] isa JuMP.ConstraintRef

        norm_container = InfrastructureOptimizationModels.get_constraint(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostNormalizationConstraint(),
            MockThermalGen,
        )
        @test norm_container["gen1", 1] isa JuMP.ConstraintRef
    end

    @testset "_get_pwl_cost_expression computes correct expression" begin
        (; container, device, pwl_data) = setup_pwl_test(; time_steps = 1:1)

        InfrastructureOptimizationModels._add_pwl_variables!(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
        )

        cost_expr = InfrastructureOptimizationModels._get_pwl_cost_expression(
            container,
            device,
            1,
            pwl_data,
            1.0,
        )

        @test cost_expr isa JuMP.AffExpr

        # Verify coefficients match y_coords
        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
            MockThermalGen,
        )
        y_coords = IS.get_y_coords(pwl_data)
        for (i, y) in enumerate(y_coords)
            var = pwl_var_container["gen1", i, 1]
            @test JuMP.coefficient(cost_expr, var) ≈ y
        end
    end

    @testset "_get_pwl_cost_expression with multiplier" begin
        (; container, device, pwl_data) = setup_pwl_test(; time_steps = 1:1)

        InfrastructureOptimizationModels._add_pwl_variables!(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
        )

        multiplier = 2.5
        cost_expr = InfrastructureOptimizationModels._get_pwl_cost_expression(
            container,
            device,
            1,
            pwl_data,
            multiplier,
        )

        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
            MockThermalGen,
        )
        y_coords = IS.get_y_coords(pwl_data)
        for (i, y) in enumerate(y_coords)
            var = pwl_var_container["gen1", i, 1]
            @test JuMP.coefficient(cost_expr, var) ≈ y * multiplier
        end
    end

    @testset "add_pwl_sos2_constraint! adds SOS2 constraint for non-convex curves" begin
        (; container, device, pwl_data) =
            setup_pwl_test(; time_steps = 1:1, points = NONCONVEX_PWL_POINTS)

        @test !IS.is_convex(pwl_data)

        pwl_vars = InfrastructureOptimizationModels._add_pwl_variables!(
            container,
            MockThermalGen,
            "gen1",
            1,
            pwl_data,
        )

        IOM.add_pwl_sos2_constraint!(container, MockThermalGen, "gen1", 1, pwl_vars)

        jump_model = InfrastructureOptimizationModels.get_jump_model(container)
        num_sos2 =
            JuMP.num_constraints(jump_model, Vector{JuMP.VariableRef}, MOI.SOS2{Float64})
        @test num_sos2 == 1
    end

    @testset "add_variable_cost_to_objective! with CostCurve{PiecewisePointCurve}" begin
        # Points in natural units (MW, $): (0, 0), (50, 1000), (100, 2500)
        natural_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 1000.0),
            (x = 100.0, y = 2500.0),
        ]

        @testset "NATURAL_UNITS" begin
            (; container, device, cost_curve) = setup_pwl_test(;
                device_base_power = 50.0,
                points = natural_points,
                unit_system = IS.UnitSystem.NATURAL_UNITS,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable(),
                device,
                cost_curve,
                TestPWLFormulation(),
            )

            # Verify PWL variables were created
            pwl_var_container = InfrastructureOptimizationModels.get_variable(
                container,
                InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
                MockThermalGen,
            )
            @test !isnothing(pwl_var_container)

            # NATURAL_UNITS: x_coords / base_power, y_coords unchanged
            # For 100 MW system base, points become (0, 0), (0.5, 1000), (1.0, 2500)
            # dt = 1.0
            obj = InfrastructureOptimizationModels.get_objective_expression(container)
            invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)
            @test length(invariant.terms) > 0

            var_y0 = pwl_var_container["gen1", 1, 1]
            var_y1000 = pwl_var_container["gen1", 2, 1]
            @test JuMP.coefficient(invariant, var_y0) ≈ 0.0 atol = 1e-10
            @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        end

        @testset "SYSTEM_BASE" begin
            # Points already in system base units (p.u., $)
            system_base_points = [
                (x = 0.0, y = 0.0),
                (x = 0.5, y = 1000.0),
                (x = 1.0, y = 2500.0),
            ]
            (; container, device, cost_curve) = setup_pwl_test(;
                device_base_power = 50.0,
                points = system_base_points,
                unit_system = IS.UnitSystem.SYSTEM_BASE,
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable(),
                device,
                cost_curve,
                TestPWLFormulation(),
            )

            # SYSTEM_BASE: no conversion needed
            obj = InfrastructureOptimizationModels.get_objective_expression(container)
            invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

            pwl_var_container = InfrastructureOptimizationModels.get_variable(
                container,
                InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
                MockThermalGen,
            )
            var_y1000 = pwl_var_container["gen1", 2, 1]
            @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        end

        @testset "with non-unity resolution (15 min)" begin
            linear_points = [(x = 0.0, y = 0.0), (x = 100.0, y = 2000.0)]
            (; container, device, cost_curve) = setup_pwl_test(;
                time_steps = 1:4,
                points = linear_points,
                resolution = Dates.Minute(15),
            )

            InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable(),
                device,
                cost_curve,
                TestPWLFormulation(),
            )

            # dt = 15/60 = 0.25 hours, y=2000 * 0.25 = 500
            obj = InfrastructureOptimizationModels.get_objective_expression(container)
            invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

            pwl_var_container = InfrastructureOptimizationModels.get_variable(
                container,
                InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
                MockThermalGen,
            )
            var_y2000 = pwl_var_container["gen1", 2, 1]
            @test JuMP.coefficient(invariant, var_y2000) ≈ 500.0 atol = 1e-10
        end

        @testset "non-convex curve adds SOS2" begin
            # Non-convex: slopes decrease (40, then 10)
            nonconvex_natural_points = [
                (x = 0.0, y = 0.0),
                (x = 50.0, y = 2000.0),
                (x = 100.0, y = 2500.0),
            ]
            (; container, device, cost_curve, pwl_data) = setup_pwl_test(;
                time_steps = 1:1,
                points = nonconvex_natural_points,
            )
            @test !IS.is_convex(pwl_data)

            @test_logs (:warn, r"not compatible with a linear PWL") InfrastructureOptimizationModels.add_variable_cost_to_objective!(
                container,
                TestPWLVariable(),
                device,
                cost_curve,
                TestPWLFormulation(),
            )

            jump_model = InfrastructureOptimizationModels.get_jump_model(container)
            num_sos2 = JuMP.num_constraints(
                jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            )
            @test num_sos2 == 1
        end
    end

    @testset "add_variable_cost_to_objective! with FuelCurve{PiecewisePointCurve}" begin
        # Fuel curve: heat rate (MMBTU/h) vs power (MW)
        # Points: (0 MW, 0), (50 MW, 400), (100 MW, 900) MMBTU/h
        fuel_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 400.0),
            (x = 100.0, y = 900.0),
        ]
        (; container, device, cost_curve) = setup_pwl_test(;
            points = fuel_points,
            fuel_cost = 5.0,  # $/MMBTU
        )

        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable(),
            device,
            cost_curve,
            TestPWLFormulation(),
        )

        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

        # Fuel consumption * fuel_cost * dt
        # y=400 * 5.0 * 1.0 = $2000, y=900 * 5.0 * 1.0 = $4500
        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
            MockThermalGen,
        )
        var_y400 = pwl_var_container["gen1", 2, 1]
        var_y900 = pwl_var_container["gen1", 3, 1]
        @test JuMP.coefficient(invariant, var_y400) ≈ 2000.0 atol = 1e-10
        @test JuMP.coefficient(invariant, var_y900) ≈ 4500.0 atol = 1e-10
    end

    @testset "add_variable_cost_to_objective! with PiecewiseIncrementalCurve" begin
        device = make_mock_thermal("gen1"; base_power = 100.0)
        container = setup_pwl_container_with_variables(1:2, device)

        # Incremental curve: marginal costs at each segment
        # x_coords: [0, 50, 100] MW, slopes: [20, 30] $/MWh
        # Converts to points: (0, 0), (50, 1000), (100, 2500)
        incremental_curve = IS.PiecewiseIncrementalCurve(
            0.0,                    # initial_input
            [0.0, 50.0, 100.0],     # x_coords
            [20.0, 30.0],           # slopes
        )
        cost_curve = IS.CostCurve(
            incremental_curve,  # Already an IncrementalCurve
            IS.UnitSystem.NATURAL_UNITS,
        )

        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable(),
            device,
            cost_curve,
            TestPWLFormulation(),
        )

        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
            MockThermalGen,
        )
        var_y1000 = pwl_var_container["gen1", 2, 1]
        var_y2500 = pwl_var_container["gen1", 3, 1]
        @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        @test JuMP.coefficient(invariant, var_y2500) ≈ 2500.0 atol = 1e-10
    end

    @testset "linear PWL (convex but not strictly convex)" begin
        # A PWL with constant slopes is convex but not strictly convex
        # This tests the edge case where is_convex returns true but slopes are equal
        # Points: (0, 0), (50, 1000), (100, 2000) - slope is constant 20 $/MWh
        linear_pwl_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 1000.0),
            (x = 100.0, y = 2000.0),
        ]
        (; container, device, cost_curve, pwl_data) = setup_pwl_test(;
            time_steps = 1:1,
            points = linear_pwl_points,
        )

        # Should be convex (equal slopes count as convex)
        @test IS.is_convex(pwl_data)

        # Should NOT add SOS2 constraint (no warning expected)
        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable(),
            device,
            cost_curve,
            TestPWLFormulation(),
        )

        # Verify no SOS2 constraint was added
        jump_model = InfrastructureOptimizationModels.get_jump_model(container)
        num_sos2 =
            JuMP.num_constraints(jump_model, Vector{JuMP.VariableRef}, MOI.SOS2{Float64})
        @test num_sos2 == 0

        # Verify costs are correct
        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)

        pwl_var_container = InfrastructureOptimizationModels.get_variable(
            container,
            InfrastructureOptimizationModels.PiecewiseLinearCostVariable(),
            MockThermalGen,
        )
        var_y1000 = pwl_var_container["gen1", 2, 1]
        var_y2000 = pwl_var_container["gen1", 3, 1]
        @test JuMP.coefficient(invariant, var_y1000) ≈ 1000.0 atol = 1e-10
        @test JuMP.coefficient(invariant, var_y2000) ≈ 2000.0 atol = 1e-10
    end

    @testset "zero cost PWL is handled correctly" begin
        zero_cost_points = [
            (x = 0.0, y = 0.0),
            (x = 50.0, y = 0.0),
            (x = 100.0, y = 0.0),
        ]
        (; container, device, cost_curve) = setup_pwl_test(; points = zero_cost_points)

        # Should return early without adding to objective
        InfrastructureOptimizationModels.add_variable_cost_to_objective!(
            container,
            TestPWLVariable(),
            device,
            cost_curve,
            TestPWLFormulation(),
        )

        obj = InfrastructureOptimizationModels.get_objective_expression(container)
        invariant = InfrastructureOptimizationModels.get_invariant_terms(obj)
        @test length(invariant.terms) == 0
    end
end
