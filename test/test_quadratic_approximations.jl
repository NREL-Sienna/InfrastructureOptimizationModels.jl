const MOI = JuMP.MOI

function _setup_qa_container(time_steps::UnitRange{Int})
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, time_steps)
    return container
end

function _setup_qa_test(device_names::Vector{String}, time_steps::UnitRange{Int})
    container = _setup_qa_container(time_steps)
    var_container = IOM.add_variable_container!(
        container,
        TestOriginalVariable(),
        MockThermalGen,
        device_names,
        time_steps,
    )
    jump_model = IOM.get_jump_model(container)
    for name in device_names, t in time_steps
        var_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "x_$(name)_$(t)",
        )
    end
    return (; container, var_container, jump_model)
end

@testset "Quadratic Approximations" begin
    @testset "Solver SOS2" begin
        @testset "Constraint structure" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            num_segments = 4
            n_points = num_segments + 1

            result = IOM._add_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, num_segments,
            )
            x_sq = result[("dev1", 1)]

            # Returned expression should be AffExpr
            @test x_sq isa JuMP.AffExpr

            # Lambda variables should exist
            lambda_container = IOM.get_variable(
                setup.container, IOM.QuadraticApproxVariable(), MockThermalGen,
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
            )

            # Normalization constraint should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.QuadraticApproxNormalizationConstraint,
                MockThermalGen,
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

            result = IOM._add_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, 4,
            )
            x_sq = result[("dev1", 1)]

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

            result = IOM._add_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, 4,
            )
            x_sq = result[("dev1", 1)]

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
            result = IOM._add_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:3, setup.var_container,
                0.0, 4.0, 4,
            )

            # Verify lambda variables exist for each time step
            lambda_container = IOM.get_variable(
                setup.container, IOM.QuadraticApproxVariable(), MockThermalGen,
            )
            for t in 1:3, i in 1:5
                @test haskey(lambda_container, ("dev1", i, t))
            end

            # Result dict should have entries for all (name, t) pairs
            for t in 1:3
                @test haskey(result, ("dev1", t))
            end
        end
    end

    @testset "Manual SOS2" begin
        @testset "Constraint structure" begin
            setup = _setup_qa_test(["dev1"], 1:1)
            num_segments = 4
            n_points = num_segments + 1

            result = IOM._add_manual_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, num_segments,
            )
            x_sq = result[("dev1", 1)]

            # Returned expression should be AffExpr
            @test x_sq isa JuMP.AffExpr

            # Lambda variables should exist
            lambda_container = IOM.get_variable(
                setup.container, IOM.QuadraticApproxVariable(), MockThermalGen,
            )
            for i in 1:n_points
                @test haskey(lambda_container, ("dev1", i, 1))
            end

            # Binary z variables should exist (n_points - 1)
            z_container = IOM.get_variable(
                setup.container, IOM.ManualSOS2BinaryVariable(), MockThermalGen,
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

            result = IOM._add_manual_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, 4,
            )
            x_sq = result[("dev1", 1)]

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

            result = IOM._add_manual_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, 4,
            )
            x_sq = result[("dev1", 1)]

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

            result = IOM._add_sawtooth_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, depth,
            )

            # Returned dict should contain AffExpr for each (name, t)
            @test haskey(result, ("dev1", 1))
            @test result[("dev1", 1)] isa JuMP.AffExpr

            # Auxiliary variables g_0, g_1, g_2 should exist
            g_container = IOM.get_variable(
                setup.container, IOM.SawtoothAuxVariable(), MockThermalGen,
            )
            for j in 0:depth
                var = g_container["dev1", j, 1]
                @test JuMP.lower_bound(var) == 0.0
                @test JuMP.upper_bound(var) == 1.0
            end

            # Binary variables α_1, α_2 should exist
            alpha_container = IOM.get_variable(
                setup.container, IOM.SawtoothBinaryVariable(), MockThermalGen,
            )
            for j in 1:depth
                @test JuMP.is_binary(alpha_container["dev1", j, 1])
            end

            # Linking constraint should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.SawtoothLinkingConstraint,
                MockThermalGen,
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

            result = IOM._add_sawtooth_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, 2,
            )
            x_sq = result[("dev1", 1)]

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

            result = IOM._add_sawtooth_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 4.0, 2,
            )
            x_sq = result[("dev1", 1)]

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
            result = IOM._add_sawtooth_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:3, setup.var_container,
                0.0, 4.0, 2,
            )

            # Verify variables exist for each time step
            g_container = IOM.get_variable(
                setup.container, IOM.SawtoothAuxVariable(), MockThermalGen,
            )
            alpha_container = IOM.get_variable(
                setup.container, IOM.SawtoothBinaryVariable(), MockThermalGen,
            )
            for t in 1:3, j in 0:2
                @test JuMP.lower_bound(g_container["dev1", j, t]) == 0.0
            end
            for t in 1:3, j in 1:2
                @test JuMP.is_binary(alpha_container["dev1", j, t])
            end

            # Result dict should have entries for all (name, t) pairs
            for t in 1:3
                @test haskey(result, ("dev1", t))
            end
        end

        @testset "Approximation quality improves with depth" begin
            # min x^2 - 6x on [0, 6], analytic minimum at x=3, value=-9
            analytic_min = -9.0
            errors = Float64[]
            for depth in [1, 2, 3, 4]
                setup = _setup_qa_test(["dev1"], 1:1)
                x_var = setup.var_container["dev1", 1]
                JuMP.set_lower_bound(x_var, 0.0)
                JuMP.set_upper_bound(x_var, 6.0)

                result = IOM._add_sawtooth_quadratic_approx!(
                    setup.container, MockThermalGen,
                    ["dev1"], 1:1, setup.var_container,
                    0.0, 6.0, depth,
                )
                x_sq = result[("dev1", 1)]

                JuMP.@objective(setup.jump_model, Min, x_sq - 6.0 * x_var)
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
            # depth=2 → 5 breakpoints on [0,4], same as SOS2 with 4 segments
            # Both should give exact result at x=2 for min x^2 - 4x
            for method in [:sos2, :sawtooth]
                setup = _setup_qa_test(["dev1"], 1:1)
                x_var = setup.var_container["dev1", 1]
                JuMP.set_lower_bound(x_var, 0.0)
                JuMP.set_upper_bound(x_var, 4.0)

                if method == :sos2
                    result = IOM._add_sos2_quadratic_approx!(
                        setup.container, MockThermalGen,
                        ["dev1"], 1:1, setup.var_container,
                        0.0, 4.0, 4,
                    )
                    x_sq = result[("dev1", 1)]
                else
                    result = IOM._add_sawtooth_quadratic_approx!(
                        setup.container, MockThermalGen,
                        ["dev1"], 1:1, setup.var_container,
                        0.0, 4.0, 2,
                    )
                    x_sq = result[("dev1", 1)]
                end

                JuMP.@objective(setup.jump_model, Min, x_sq - 4.0 * x_var)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                @test JuMP.objective_value(setup.jump_model) ≈ -4.0 atol = 1e-6
            end
        end
    end

    @testset "Approximation quality improves with more segments" begin
        # min x^2 - 6x on [0, 6], analytic minimum at x=3, value=-9
        analytic_min = -9.0
        errors = Float64[]
        for num_segments in [2, 4, 8, 16]
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 6.0)

            result = IOM._add_sos2_quadratic_approx!(
                setup.container, MockThermalGen,
                ["dev1"], 1:1, setup.var_container,
                0.0, 6.0, num_segments,
            )
            x_sq = result[("dev1", 1)]

            JuMP.@objective(setup.jump_model, Min, x_sq - 6.0 * x_var)
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
