const MOI = JuMP.MOI
const TEST_META = "TestVar"

@testset "Quadratic Approximations" begin
    @testset "Solver SOS2" begin
        @testset "Constraint structure" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            num_segments = 4
            n_points = num_segments + 1

            IOM._add_sos2_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                num_segments,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            # Returned expression should be AffExpr
            @test x_sq isa JuMP.AffExpr

            # Lambda variables should exist
            lambda_container = IOM.get_variable(
                setup.container,
                IOM.QuadraticApproxVariable(),
                MockThermalGen,
                TEST_META,
            )
            for i in 1:n_points
                @test haskey(lambda_container, ("dev1", i, 1))
                var = lambda_container[("dev1", i, 1)]
                @test JuMP.lower_bound(var) == 0.0
                @test JuMP.upper_bound(var) == 1.0
            end

            # Linking constraint should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.QuadraticApproxLinkingConstraint,
                MockThermalGen,
                TEST_META,
            )

            # Normalization constraint should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.QuadraticApproxNormalizationConstraint,
                MockThermalGen,
                TEST_META,
            )

            # SOS2 constraint should exist (solver-native)
            sos2_count = JuMP.num_constraints(
                setup.jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            )
            @test sos2_count == 1
        end

        @testset "Solve min x^2 - 4x" begin
            # Analytic minimum of x^2 - 4x at x=2, value = -4
            # With breakpoints at 0,1,2,3,4 the approximation is exact at breakpoints
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)

            IOM._add_sos2_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                4,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            # Objective: x^2 - 4x
            JuMP.@objective(setup.jump_model, Min, x_sq - 4.0 * x_var)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(x_var) ≈ 2.0 atol = 1e-6
            @test JuMP.objective_value(setup.jump_model) ≈ -4.0 atol = 1e-6
        end

        @testset "Constraint usage: x^2 + y = 10 with x=3" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)

            y = JuMP.@variable(setup.jump_model, base_name = "y")

            IOM._add_sos2_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                4,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            # x^2 + y = 10 → with x=3, x^2=9, y=1
            JuMP.@constraint(setup.jump_model, x_sq + y == 10.0)
            JuMP.@objective(setup.jump_model, Min, y)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(y) ≈ 1.0 atol = 1e-6
        end

        @testset "Multiple time steps" begin
            setup = _setup_qa_test(["dev1"], 1:3)
            IOM._add_sos2_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:3,
                setup.var_container,
                0.0,
                4.0,
                4,
                TEST_META,
            )

            # Verify lambda variables exist for each time step
            lambda_container = IOM.get_variable(
                setup.container,
                IOM.QuadraticApproxVariable(),
                MockThermalGen,
                TEST_META,
            )
            for t in 1:3, i in 1:5
                @test haskey(lambda_container, ("dev1", i, t))
            end

            # Expression container should have entries for all (name, t) pairs
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            for t in 1:3
                @test expr_container["dev1", t] isa JuMP.AffExpr
            end
        end

        @testset "Approximation quality improves with more segments" begin
            # min (√2)x² - (√3)x on [0, 6], analytic minimum at x=√3/8
            analytic_min = sqrt(2) * (sqrt(3 / 8)^2) - sqrt(3) * sqrt(3 / 8)
            errors = Float64[]
            for num_segments in 2 .^ (1:6)
                setup = _setup_qa_test(["dev1"], 1:1)
                x_var = setup.var_container["dev1", 1]
                JuMP.set_lower_bound(x_var, 0.0)
                JuMP.set_upper_bound(x_var, 6.0)

                IOM._add_sos2_quadratic_approx!(
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.var_container,
                    0.0,
                    6.0,
                    num_segments,
                    TEST_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.QuadraticApproximationExpression(),
                    MockThermalGen,
                    TEST_META,
                )
                x_sq = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Min, sqrt(2) * x_sq - sqrt(3) * x_var)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - analytic_min))
            end
            # Each doubling of segments should reduce error (or maintain if already exact)
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end

    @testset "Manual SOS2" begin
        @testset "Constraint structure" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            num_segments = 4
            n_points = num_segments + 1

            IOM._add_manual_sos2_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                num_segments,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            # Returned expression should be AffExpr
            @test x_sq isa JuMP.AffExpr

            # Lambda variables should exist
            lambda_container = IOM.get_variable(
                setup.container,
                IOM.QuadraticApproxVariable(),
                MockThermalGen,
                TEST_META,
            )
            for i in 1:n_points
                @test haskey(lambda_container, ("dev1", i, 1))
            end

            # Binary z variables should exist (n_points - 1)
            z_container = IOM.get_variable(
                setup.container,
                IOM.ManualSOS2BinaryVariable(),
                MockThermalGen,
                TEST_META,
            )
            for j in 1:(n_points - 1)
                @test haskey(z_container, ("dev1", j, 1))
                @test JuMP.is_binary(z_container[("dev1", j, 1)])
            end

            # Segment selection constraint should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.ManualSOS2SegmentSelectionConstraint,
                MockThermalGen,
                TEST_META,
            )

            # NO solver SOS2 constraints
            sos2_count = JuMP.num_constraints(
                setup.jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            )
            @test sos2_count == 0
        end

        @testset "Solve min x^2 - 4x" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)

            IOM._add_manual_sos2_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                4,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, x_sq - 4.0 * x_var)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(x_var) ≈ 2.0 atol = 1e-6
            @test JuMP.objective_value(setup.jump_model) ≈ -4.0 atol = 1e-6
        end

        @testset "Constraint usage: x^2 + y = 10 with x=3" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)

            y = JuMP.@variable(setup.jump_model, base_name = "y")

            IOM._add_manual_sos2_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                4,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@constraint(setup.jump_model, x_sq + y == 10.0)
            JuMP.@objective(setup.jump_model, Min, y)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(y) ≈ 1.0 atol = 1e-6
        end
    end

    @testset "Sawtooth" begin
        @testset "Constraint structure" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            depth = 2

            IOM._add_sawtooth_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                depth,
                TEST_META,
            )

            # Expression container should contain AffExpr for each (name, t)
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            @test expr_container["dev1", 1] isa JuMP.AffExpr

            # Auxiliary variables g_0, g_1, g_2 should exist
            g_container = IOM.get_variable(
                setup.container,
                IOM.SawtoothAuxVariable(),
                MockThermalGen,
                TEST_META,
            )
            for j in 0:depth
                var = g_container["dev1", j, 1]
                @test JuMP.lower_bound(var) == 0.0
                @test JuMP.upper_bound(var) == 1.0
            end

            # Binary variables α_1, α_2 should exist
            alpha_container = IOM.get_variable(
                setup.container,
                IOM.SawtoothBinaryVariable(),
                MockThermalGen,
                TEST_META,
            )
            for j in 1:depth
                @test JuMP.is_binary(alpha_container["dev1", j, 1])
            end

            # Linking constraint should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.SawtoothLinkingConstraint,
                MockThermalGen,
                TEST_META,
            )

            # NO solver SOS2 constraints
            sos2_count = JuMP.num_constraints(
                setup.jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            )
            @test sos2_count == 0
        end

        @testset "Solve min x^2 - 4x" begin
            # depth=2 → breakpoints at 0,1,2,3,4 → exact at x=2
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)

            IOM._add_sawtooth_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                2,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, x_sq - 4.0 * x_var)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(x_var) ≈ 2.0 atol = 1e-6
            @test JuMP.objective_value(setup.jump_model) ≈ -4.0 atol = 1e-6
        end

        @testset "Constraint usage: x^2 + y = 10 with x=3" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)

            y = JuMP.@variable(setup.jump_model, base_name = "y")

            IOM._add_sawtooth_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                4.0,
                2,
                TEST_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            x_sq = expr_container["dev1", 1]

            JuMP.@constraint(setup.jump_model, x_sq + y == 10.0)
            JuMP.@objective(setup.jump_model, Min, y)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(y) ≈ 1.0 atol = 1e-6
        end

        @testset "Multiple time steps" begin
            setup = _setup_qa_test(["dev1"], 1:3)
            IOM._add_sawtooth_quadratic_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:3,
                setup.var_container,
                0.0,
                4.0,
                2,
                TEST_META,
            )

            # Verify variables exist for each time step
            g_container = IOM.get_variable(
                setup.container,
                IOM.SawtoothAuxVariable(),
                MockThermalGen,
                TEST_META,
            )
            alpha_container = IOM.get_variable(
                setup.container,
                IOM.SawtoothBinaryVariable(),
                MockThermalGen,
                TEST_META,
            )
            for t in 1:3, j in 0:2
                @test JuMP.lower_bound(g_container["dev1", j, t]) == 0.0
            end
            for t in 1:3, j in 1:2
                @test JuMP.is_binary(alpha_container["dev1", j, t])
            end

            # Expression container should have entries for all (name, t) pairs
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticApproximationExpression(),
                MockThermalGen,
                TEST_META,
            )
            for t in 1:3
                @test expr_container["dev1", t] isa JuMP.AffExpr
            end
        end

        @testset "Approximation quality improves with depth" begin
            # min (√2)x² - (√3)x on [0, 6], analytic minimum at x=√3/8
            analytic_min = sqrt(2) * (sqrt(3 / 8)^2) - sqrt(3) * sqrt(3 / 8)
            errors = Float64[]
            for depth in 1:6
                setup = _setup_qa_test(["dev1"], 1:1)
                x_var = setup.var_container["dev1", 1]
                JuMP.set_lower_bound(x_var, 0.0)
                JuMP.set_upper_bound(x_var, 6.0)

                IOM._add_sawtooth_quadratic_approx!(
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.var_container,
                    0.0,
                    6.0,
                    depth,
                    TEST_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.QuadraticApproximationExpression(),
                    MockThermalGen,
                    TEST_META,
                )
                x_sq = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, Min, sqrt(2) * x_sq - sqrt(3) * x_var)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - analytic_min))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end

        @testset "Agrees with SOS2 at aligned breakpoints" begin
            # SOS2 and Sawtooth should agree when n_sos2_segments = 2^(sawtooth_depth)
            for depth in 1:6
                sos2_value = nothing
                for method in [:sos2, :sawtooth]
                    setup = _setup_qa_test(["dev1"], 1:1)
                    x_var = setup.var_container["dev1", 1]
                    JuMP.set_lower_bound(x_var, 0.0)
                    JuMP.set_upper_bound(x_var, 4.0)

                    if method == :sos2
                        IOM._add_sos2_quadratic_approx!(
                            setup.container,
                            MockThermalGen,
                            ["dev1"],
                            1:1,
                            setup.var_container,
                            0.0,
                            4.0,
                            2^depth,
                            TEST_META,
                        )
                    else
                        IOM._add_sawtooth_quadratic_approx!(
                            setup.container,
                            MockThermalGen,
                            ["dev1"],
                            1:1,
                            setup.var_container,
                            0.0,
                            4.0,
                            depth,
                            TEST_META,
                        )
                    end
                    expr_container = IOM.get_expression(
                        setup.container,
                        IOM.QuadraticApproximationExpression(),
                        MockThermalGen,
                        TEST_META,
                    )
                    x_sq = expr_container["dev1", 1]

                    JuMP.@objective(setup.jump_model, Min, sqrt(2) * x_sq - sqrt(3) * x_var)
                    JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                    JuMP.set_silent(setup.jump_model)
                    JuMP.optimize!(setup.jump_model)

                    if method == :sos2
                        sos2_value = JuMP.objective_value(setup.jump_model)
                    else
                        sawtooth_value = JuMP.objective_value(setup.jump_model)
                        @test sos2_value ≈ sawtooth_value atol = 1e-6
                    end
                end
            end
        end
    end
end
