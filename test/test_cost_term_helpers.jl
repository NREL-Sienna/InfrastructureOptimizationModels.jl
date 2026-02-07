"""
Unit tests for cost term helpers in src/objective_function/cost_term_helpers.jl.
Tests the generic building blocks for adding costs to expressions and objectives.

Uses common helpers from test_utils/objective_function_helpers.jl:
- make_test_container, add_test_variable!, add_test_expression!, add_test_parameter!
Test types defined in test_utils/test_types.jl.
"""

@testset "Cost Term Helpers" begin
    @testset "add_cost_term_invariant!" begin
        @testset "adds cost to invariant objective" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)

            rate = 10.0
            cost = IOM.add_cost_term_invariant!(
                container, var, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            # Verify cost expression is var * rate
            @test cost == var * rate

            # Verify it was added to invariant objective
            obj = IOM.get_objective_expression(container)
            invariant = IOM.get_invariant_terms(obj)
            @test JuMP.coefficient(invariant, var) ≈ rate

            # Verify variant is empty
            variant = IOM.get_variant_terms(obj)
            @test JuMP.coefficient(variant, var) ≈ 0.0
        end

        @testset "adds cost to expression if present" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)
            add_test_expression!(
                container,
                TestCostExpression,
                MockThermalGen,
                ["gen1"],
                1:3,
            )

            rate = 15.0
            IOM.add_cost_term_invariant!(
                container, var, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            # Verify expression was updated
            expr_container =
                IOM.get_expression(container, TestCostExpression(), MockThermalGen)
            expr = expr_container["gen1", 1]
            @test JuMP.coefficient(expr, var) ≈ rate
        end

        @testset "skips expression if not present" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)
            # Don't add expression container

            rate = 10.0
            # Should not error even without expression container
            cost = IOM.add_cost_term_invariant!(
                container, var, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            @test cost == var * rate
        end

        @testset "handles zero rate" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)

            cost = IOM.add_cost_term_invariant!(
                container, var, 0.0, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            @test cost == 0.0
        end

        @testset "handles scalar quantity (Float64)" begin
            container = make_test_container(1:3)
            add_test_expression!(
                container,
                TestCostExpression,
                MockThermalGen,
                ["gen1"],
                1:3,
            )

            quantity = 5.0
            rate = 10.0
            cost = IOM.add_cost_term_invariant!(
                container, quantity, rate, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            @test cost ≈ 50.0

            # Verify constant was added to objective
            obj = IOM.get_objective_expression(container)
            invariant = IOM.get_invariant_terms(obj)
            @test JuMP.constant(invariant) ≈ 50.0
        end
    end

    #= TODO: add_cost_term_variant! tests require parameter container infrastructure
    @testset "add_cost_term_variant!" begin
        @testset "adds cost to variant objective using parameter rate" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)

            # Add parameter with value 20.0
            param_values = [20.0 20.0 20.0]  # 1 device, 3 time steps
            add_test_parameter!(
                container, TestCostParameter, MockThermalGen, ["gen1"], 1:3, param_values
            )

            cost = IOM.add_cost_term_variant!(
                container, var, TestCostParameter, TestCostExpression, MockThermalGen, "gen1", 1
            )

            # Verify it was added to variant objective (coefficient should be parameter * multiplier = 20)
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)
            @test JuMP.coefficient(variant, var) ≈ 20.0

            # Verify invariant is empty
            invariant = IOM.get_invariant_terms(obj)
            @test JuMP.coefficient(invariant, var) ≈ 0.0
        end

        @testset "adds cost to expression if present" begin
            container = make_test_container(1:3)
            var = add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)
            add_test_expression!(container, TestCostExpression, MockThermalGen, ["gen1"], 1:3)
            param_values = [15.0 15.0 15.0]
            add_test_parameter!(
                container, TestCostParameter, MockThermalGen, ["gen1"], 1:3, param_values
            )

            IOM.add_cost_term_variant!(
                container, var, TestCostParameter, TestCostExpression, MockThermalGen, "gen1", 1
            )

            # Verify expression was updated
            expr_container = IOM.get_expression(container, TestCostExpression(), MockThermalGen)
            expr = expr_container["gen1", 1]
            @test JuMP.coefficient(expr, var) ≈ 15.0
        end
    end
    =#

    @testset "PWL Helpers" begin
        @testset "add_pwl_variables! creates bounded variables" begin
            container = make_test_container(1:3)

            pwl_vars = IOM.add_pwl_variables!(
                container, TestPWLVariable, MockThermalGen, "gen1", 1, 4,
            )

            @test length(pwl_vars) == 4
            for (i, var) in enumerate(pwl_vars)
                @test JuMP.lower_bound(var) == 0.0
                @test JuMP.upper_bound(var) == 1.0
                # Check name contains key elements (type names may be fully qualified)
                # FIXME variable names depend on from where the function is called. ick.
                var_name = JuMP.name(var)
                @test occursin("TestPWLVariable", var_name)
                @test occursin("gen1", var_name)
                @test occursin("pwl_$(i)", var_name)
            end

            # Verify stored in container
            var_container = IOM.get_variable(container, TestPWLVariable(), MockThermalGen)
            for i in 1:4
                @test var_container["gen1", i, 1] === pwl_vars[i]
            end
        end

        @testset "add_pwl_variables! with custom upper bound" begin
            container = make_test_container(1:3)

            pwl_vars = IOM.add_pwl_variables!(
                container, TestPWLVariable, MockThermalGen, "gen1", 1, 3;
                upper_bound = 100.0,
            )

            @test length(pwl_vars) == 3
            for var in pwl_vars
                @test JuMP.lower_bound(var) == 0.0
                @test JuMP.upper_bound(var) == 100.0
            end
        end

        @testset "add_pwl_variables! with no upper bound (Inf)" begin
            container = make_test_container(1:3)

            pwl_vars = IOM.add_pwl_variables!(
                container, TestPWLVariable, MockThermalGen, "gen1", 1, 3;
                upper_bound = Inf,
            )

            @test length(pwl_vars) == 3
            for var in pwl_vars
                @test JuMP.lower_bound(var) == 0.0
                @test !JuMP.has_upper_bound(var)
            end
        end

        @testset "add_pwl_linking_constraint! creates correct constraint" begin
            container = make_test_container(1:3)

            # Create power variable
            power_var =
                add_test_variable!(container, TestCostVariable, MockThermalGen, "gen1", 1)

            # Create PWL delta variables manually
            jump_model = IOM.get_jump_model(container)
            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            breakpoints = [0.0, 50.0, 100.0]

            IOM.add_pwl_linking_constraint!(
                container, TestCostConstraint, MockThermalGen, "gen1", 1,
                power_var, pwl_vars, breakpoints,
            )

            # Get the constraint
            con_container =
                IOM.get_constraint(container, TestCostConstraint(), MockThermalGen)
            con = con_container["gen1", 1]

            # Verify constraint: power_var == sum(pwl_vars .* breakpoints)
            # In normalized form: power_var - 0*δ1 - 50*δ2 - 100*δ3 == 0
            con_func = JuMP.constraint_object(con).func
            @test JuMP.coefficient(con_func, power_var) ≈ 1.0
            @test JuMP.coefficient(con_func, pwl_vars[1]) ≈ -0.0
            @test JuMP.coefficient(con_func, pwl_vars[2]) ≈ -50.0
            @test JuMP.coefficient(con_func, pwl_vars[3]) ≈ -100.0
        end

        @testset "add_pwl_normalization_constraint! creates correct constraint" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            on_status = 1.0

            IOM.add_pwl_normalization_constraint!(
                container, TestCostConstraint, MockThermalGen, "gen1", 1,
                pwl_vars, on_status,
            )

            con_container =
                IOM.get_constraint(container, TestCostConstraint(), MockThermalGen)
            con = con_container["gen1", 1]

            # Verify constraint: sum(pwl_vars) == on_status
            # In normalized form: δ1 + δ2 + δ3 == 1
            con_func = JuMP.constraint_object(con).func
            for var in pwl_vars
                @test JuMP.coefficient(con_func, var) ≈ 1.0
            end
            @test JuMP.normalized_rhs(con) ≈ 1.0
        end

        @testset "add_pwl_normalization_constraint! with variable on_status" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            on_var = JuMP.@variable(jump_model, base_name = "on_status", binary = true)

            IOM.add_pwl_normalization_constraint!(
                container, TestCostConstraint, MockThermalGen, "gen1", 1,
                pwl_vars, on_var,
            )

            con_container =
                IOM.get_constraint(container, TestCostConstraint(), MockThermalGen)
            con = con_container["gen1", 1]

            # Verify constraint includes the on_status variable
            con_func = JuMP.constraint_object(con).func
            @test JuMP.coefficient(con_func, on_var) ≈ -1.0
        end

        @testset "add_pwl_sos2_constraint! creates SOS2 constraint" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:4]

            IOM.add_pwl_sos2_constraint!(
                container, MockThermalGen, "gen1", 1, pwl_vars,
            )

            # Verify SOS2 constraint was added to the model
            # JuMP stores SOS constraints separately
            @test JuMP.num_constraints(
                jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            ) == 1
        end

        @testset "get_pwl_cost_expression computes correct expression" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:3]
            slopes = [10.0, 20.0, 30.0]
            multiplier = 2.0

            cost_expr = IOM.get_pwl_cost_expression(pwl_vars, slopes, multiplier)

            # Verify: cost = Σ δ[i] * slope[i] * multiplier
            # = δ1 * 10 * 2 + δ2 * 20 * 2 + δ3 * 30 * 2
            # = 20*δ1 + 40*δ2 + 60*δ3
            @test JuMP.coefficient(cost_expr, pwl_vars[1]) ≈ 20.0
            @test JuMP.coefficient(cost_expr, pwl_vars[2]) ≈ 40.0
            @test JuMP.coefficient(cost_expr, pwl_vars[3]) ≈ 60.0
            @test JuMP.constant(cost_expr) ≈ 0.0
        end

        @testset "get_pwl_cost_expression with multiplier = 1" begin
            container = make_test_container(1:3)
            jump_model = IOM.get_jump_model(container)

            pwl_vars = [JuMP.@variable(jump_model, base_name = "delta_$i") for i in 1:2]
            slopes = [5.0, 15.0]

            cost_expr = IOM.get_pwl_cost_expression(pwl_vars, slopes, 1.0)

            @test JuMP.coefficient(cost_expr, pwl_vars[1]) ≈ 5.0
            @test JuMP.coefficient(cost_expr, pwl_vars[2]) ≈ 15.0
        end
    end

    @testset "Integration: PWL cost added to objective" begin
        @testset "PWL cost via add_cost_term_invariant!" begin
            container = make_test_container(1:1)

            # Create PWL variables
            pwl_vars = IOM.add_pwl_variables!(
                container, TestPWLVariable, MockThermalGen, "gen1", 1, 3,
            )

            # Compute cost expression
            slopes = [0.0, 10.0, 25.0]  # costs at each breakpoint
            cost_expr = IOM.get_pwl_cost_expression(pwl_vars, slopes, 1.0)

            # Add to invariant objective
            IOM.add_cost_term_invariant!(
                container, cost_expr, 1.0, TestCostExpression, MockThermalGen, "gen1", 1,
            )

            # Verify coefficients in objective
            obj = IOM.get_objective_expression(container)
            invariant = IOM.get_invariant_terms(obj)
            @test JuMP.coefficient(invariant, pwl_vars[1]) ≈ 0.0
            @test JuMP.coefficient(invariant, pwl_vars[2]) ≈ 10.0
            @test JuMP.coefficient(invariant, pwl_vars[3]) ≈ 25.0
        end
    end
end
