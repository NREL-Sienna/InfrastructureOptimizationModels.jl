const BILINEAR_META = "BilinearTest"

function _setup_bilinear_test(device_names::Vector{String}, time_steps::UnitRange{Int})
    container = _setup_qa_container(time_steps)
    x_var_container = IOM.add_variable_container!(
        container,
        TestOriginalVariable(),
        MockThermalGen,
        device_names,
        time_steps,
    )
    y_var_container = IOM.add_variable_container!(
        container,
        TestApproximatedVariable(),
        MockThermalGen,
        device_names,
        time_steps,
    )
    jump_model = IOM.get_jump_model(container)
    for name in device_names, t in time_steps
        x_var_container[name, t] =
            JuMP.@variable(jump_model, base_name = "x_$(name)_$(t)")
        y_var_container[name, t] =
            JuMP.@variable(jump_model, base_name = "y_$(name)_$(t)")
    end
    return (; container, x_var_container, y_var_container, jump_model)
end

@testset "Bilinear Approximations" begin
    @testset "Solver SOS2 Bilinear" begin
        @testset "Constraint structure" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)

            result = IOM._add_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                4,
                BILINEAR_META,
            )

            @test haskey(result, ("dev1", 1))
            @test result[("dev1", 1)] isa JuMP.AffExpr

            # u = x + y variable container should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.BilinearApproxSumVariable,
                MockThermalGen,
                BILINEAR_META * "_plus",
            )
            # v = x - y variable container should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.BilinearApproxDiffVariable,
                MockThermalGen,
                BILINEAR_META * "_minus",
            )
            # z ≈ x·y variable container should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.BilinearProductVariable,
                MockThermalGen,
                BILINEAR_META,
            )
            # Linking constraints should exist
            @test IOM.has_container_key(
                setup.container,
                IOM.BilinearApproxSumLinkingConstraint,
                MockThermalGen,
                BILINEAR_META * "_plus",
            )
            @test IOM.has_container_key(
                setup.container,
                IOM.BilinearApproxDiffLinkingConstraint,
                MockThermalGen,
                BILINEAR_META * "_minus",
            )
            # Inner quadratic approx containers should exist with _plus/_minus meta
            @test IOM.has_container_key(
                setup.container,
                IOM.QuadraticApproxVariable,
                MockThermalGen,
                BILINEAR_META * "_plus",
            )
            @test IOM.has_container_key(
                setup.container,
                IOM.QuadraticApproxVariable,
                MockThermalGen,
                BILINEAR_META * "_minus",
            )

            # u bounds should be [0+0, 4+4] = [0, 8]
            u_container = IOM.get_variable(
                setup.container,
                IOM.BilinearApproxSumVariable(),
                MockThermalGen,
                BILINEAR_META * "_plus",
            )
            @test JuMP.lower_bound(u_container["dev1", 1]) == 0.0
            @test JuMP.upper_bound(u_container["dev1", 1]) == 8.0

            # v bounds should be [0-4, 4-0] = [-4, 4]
            v_container = IOM.get_variable(
                setup.container,
                IOM.BilinearApproxDiffVariable(),
                MockThermalGen,
                BILINEAR_META * "_minus",
            )
            @test JuMP.lower_bound(v_container["dev1", 1]) == -4.0
            @test JuMP.upper_bound(v_container["dev1", 1]) == 4.0
        end

        @testset "Constraint structure with McCormick" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)

            IOM._add_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                4,
                BILINEAR_META;
                add_mccormick = true,
            )

            @test IOM.has_container_key(
                setup.container,
                IOM.McCormickConstraint,
                MockThermalGen,
                BILINEAR_META,
            )
        end

        @testset "Fixed-variable correctness" begin
            # Fix x=3, y ∈ [0,4]: min xy should give z≈0 at y=0
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.fix(x_var, 3.0; force = true)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            result = IOM._add_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                8,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-4

            # Fix x=0, y=0: z should be exactly 0
            setup2 = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup2.x_var_container["dev1", 1], 0.0; force = true)
            JuMP.fix(setup2.y_var_container["dev1", 1], 0.0; force = true)

            result2 = IOM._add_sos2_bilinear_approx!(
                setup2.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup2.x_var_container,
                setup2.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                4,
                BILINEAR_META,
            )
            z_expr2 = result2[("dev1", 1)]

            JuMP.@objective(setup2.jump_model, Max, z_expr2)
            JuMP.set_optimizer(setup2.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup2.jump_model)
            JuMP.optimize!(setup2.jump_model)

            @test JuMP.termination_status(setup2.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup2.jump_model) ≈ 0.0 atol = 1e-6
        end

        @testset "Constraint usage: x·y + w = 10 with x=2" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.fix(x_var, 2.0; force = true)
            JuMP.fix(y_var, 3.0; force = true)

            w = JuMP.@variable(setup.jump_model, base_name = "w")

            result = IOM._add_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                8,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            # x·y + w = 10 → 2·3 + w = 10 → w = 4
            JuMP.@constraint(setup.jump_model, z_expr + w == 10.0)
            JuMP.@objective(setup.jump_model, Min, w)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(w) ≈ 4.0 atol = 1e-4
        end

        @testset "Vertex optimum" begin
            # min x·y on [0,4]×[0,4]; minimum is 0 at a corner
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            result = IOM._add_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                8,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-4
        end

        @testset "Multiple time steps" begin
            setup = _setup_bilinear_test(["dev1"], 1:3)
            result = IOM._add_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:3,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                4,
                BILINEAR_META,
            )

            for t in 1:3
                @test haskey(result, ("dev1", t))
                @test result[("dev1", t)] isa JuMP.AffExpr
            end
        end

        @testset "Approximation quality improves with segments" begin
            # Fix x=2.5, y=1.5: true product = 3.75
            # Sweep segments, verify gap shrinks
            true_product = 2.5 * 1.5
            errors = Float64[]
            for num_segments in 2 .^ (1:5)
                setup = _setup_bilinear_test(["dev1"], 1:1)
                x_var = setup.x_var_container["dev1", 1]
                y_var = setup.y_var_container["dev1", 1]
                JuMP.fix(x_var, 2.5; force = true)
                JuMP.fix(y_var, 1.5; force = true)

                result = IOM._add_sos2_bilinear_approx!(
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    0.0, 4.0,
                    0.0, 4.0,
                    num_segments,
                    BILINEAR_META,
                )
                z_expr = result[("dev1", 1)]

                JuMP.@objective(setup.jump_model, Max, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - true_product))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end

    @testset "Manual SOS2 Bilinear" begin
        @testset "Constraint structure" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)

            result = IOM._add_manual_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                4,
                BILINEAR_META,
            )

            @test haskey(result, ("dev1", 1))
            @test result[("dev1", 1)] isa JuMP.AffExpr

            # Binary variables should exist for both u² and v² paths
            @test IOM.has_container_key(
                setup.container,
                IOM.ManualSOS2BinaryVariable,
                MockThermalGen,
                BILINEAR_META * "_plus",
            )
            @test IOM.has_container_key(
                setup.container,
                IOM.ManualSOS2BinaryVariable,
                MockThermalGen,
                BILINEAR_META * "_minus",
            )

            # NO solver SOS2 constraints
            sos2_count = JuMP.num_constraints(
                setup.jump_model,
                Vector{JuMP.VariableRef},
                MOI.SOS2{Float64},
            )
            @test sos2_count == 0
        end

        @testset "Fixed-variable correctness" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            result = IOM._add_manual_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                8,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 1e-4
        end

        @testset "Constraint usage: x·y + w = 10 with x=2, y=3" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            w = JuMP.@variable(setup.jump_model, base_name = "w")

            result = IOM._add_manual_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                8,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@constraint(setup.jump_model, z_expr + w == 10.0)
            JuMP.@objective(setup.jump_model, Min, w)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(w) ≈ 4.0 atol = 1e-4
        end

        @testset "Vertex optimum" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            result = IOM._add_manual_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                8,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-4
        end

        @testset "Multiple time steps" begin
            setup = _setup_bilinear_test(["dev1"], 1:3)
            result = IOM._add_manual_sos2_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:3,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                4,
                BILINEAR_META,
            )

            for t in 1:3
                @test haskey(result, ("dev1", t))
            end
        end

        @testset "Approximation quality improves with segments" begin
            true_product = 2.5 * 1.5
            errors = Float64[]
            for num_segments in 2 .^ (1:5)
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], 2.5; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], 1.5; force = true)

                result = IOM._add_manual_sos2_bilinear_approx!(
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    0.0, 4.0,
                    0.0, 4.0,
                    num_segments,
                    BILINEAR_META,
                )
                z_expr = result[("dev1", 1)]

                JuMP.@objective(setup.jump_model, Max, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - true_product))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end

    @testset "Sawtooth Bilinear" begin
        @testset "Constraint structure" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)

            result = IOM._add_sawtooth_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                2,
                BILINEAR_META,
            )

            @test haskey(result, ("dev1", 1))
            @test result[("dev1", 1)] isa JuMP.AffExpr

            # Sawtooth aux/binary variables for both u² and v² paths
            @test IOM.has_container_key(
                setup.container,
                IOM.SawtoothAuxVariable,
                MockThermalGen,
                BILINEAR_META * "_plus",
            )
            @test IOM.has_container_key(
                setup.container,
                IOM.SawtoothBinaryVariable,
                MockThermalGen,
                BILINEAR_META * "_plus",
            )
            @test IOM.has_container_key(
                setup.container,
                IOM.SawtoothAuxVariable,
                MockThermalGen,
                BILINEAR_META * "_minus",
            )
            @test IOM.has_container_key(
                setup.container,
                IOM.SawtoothBinaryVariable,
                MockThermalGen,
                BILINEAR_META * "_minus",
            )
        end

        @testset "Fixed-variable correctness" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            result = IOM._add_sawtooth_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                3,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 1e-3
        end

        @testset "Constraint usage: x·y + w = 10 with x=2, y=3" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            w = JuMP.@variable(setup.jump_model, base_name = "w")

            result = IOM._add_sawtooth_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                3,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@constraint(setup.jump_model, z_expr + w == 10.0)
            JuMP.@objective(setup.jump_model, Min, w)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.value(w) ≈ 4.0 atol = 1e-3
        end

        @testset "Vertex optimum" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            x_var = setup.x_var_container["dev1", 1]
            y_var = setup.y_var_container["dev1", 1]
            JuMP.set_lower_bound(x_var, 0.0)
            JuMP.set_upper_bound(x_var, 4.0)
            JuMP.set_lower_bound(y_var, 0.0)
            JuMP.set_upper_bound(y_var, 4.0)

            result = IOM._add_sawtooth_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                3,
                BILINEAR_META,
            )
            z_expr = result[("dev1", 1)]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-3
        end

        @testset "Multiple time steps" begin
            setup = _setup_bilinear_test(["dev1"], 1:3)
            result = IOM._add_sawtooth_bilinear_approx!(
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:3,
                setup.x_var_container,
                setup.y_var_container,
                0.0, 4.0,
                0.0, 4.0,
                2,
                BILINEAR_META,
            )

            for t in 1:3
                @test haskey(result, ("dev1", t))
            end
        end

        @testset "Approximation quality improves with depth" begin
            true_product = 2.5 * 1.5
            errors = Float64[]
            for depth in 1:5
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], 2.5; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], 1.5; force = true)

                result = IOM._add_sawtooth_bilinear_approx!(
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    0.0, 4.0,
                    0.0, 4.0,
                    depth,
                    BILINEAR_META,
                )
                z_expr = result[("dev1", 1)]

                JuMP.@objective(setup.jump_model, Max, z_expr)
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)

                obj_val = JuMP.objective_value(setup.jump_model)
                push!(errors, abs(obj_val - true_product))
            end
            for i in 2:length(errors)
                @test errors[i] <= errors[i - 1] + 1e-10
            end
        end
    end
end
