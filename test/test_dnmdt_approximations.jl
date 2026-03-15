const DNMDT_META = "DNMDTTest"
const DNMDT_HYBS_META = "HybSTest"

@testset "D-NMDT Univariate Approximation" begin
    @testset "Binary expansion correctness" begin
        names = ["gen1"]
        ts = 1:1
        setup = _setup_qa_test(names, ts)
        JuMP.set_lower_bound(setup.var_container["gen1", 1], 0.0)
        JuMP.set_upper_bound(setup.var_container["gen1", 1], 1.0)
        JuMP.fix(setup.var_container["gen1", 1], 0.6; force = true)

        IOM._add_dnmdt_univariate_approx!(
            setup.container, MockThermalGen, names, ts,
            setup.var_container, 0.0, 1.0, 4, DNMDT_META,
        )

        JuMP.@objective(setup.jump_model, Min, 0)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL

        xh = IOM.get_expression(
            setup.container, IOM.DNMDTScaledVariableExpression(), MockThermalGen,
            DNMDT_META,
        )
        beta = IOM.get_variable(
            setup.container, IOM.DNMDTBinaryVariable(), MockThermalGen, DNMDT_META,
        )
        dx = IOM.get_variable(
            setup.container, IOM.DNMDTResidualVariable(), MockThermalGen, DNMDT_META,
        )

        @test JuMP.value(xh["gen1", 1]) ≈ 0.6 atol = 1e-8
        reconstructed =
            sum(2.0^(-j) * JuMP.value(beta["gen1", j, 1]) for j in 1:4) +
            JuMP.value(dx["gen1", 1])
        @test reconstructed ≈ 0.6 atol = 1e-8
    end

    @testset "Relaxation validity (D-NMDT)" begin
        test_points = [0.1, 0.3, 0.5, 0.7, 0.9]
        for x0 in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_qa_test(["gen1"], 1:1)
                JuMP.fix(setup.var_container["gen1", 1], x0; force = true)

                IOM._add_dnmdt_univariate_approx!(
                    setup.container, MockThermalGen, ["gen1"], 1:1,
                    setup.var_container, 0.0, 1.0, 3, DNMDT_META;
                    tighten = false,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.QuadraticExpression(),
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["gen1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                push!(z_vals, JuMP.objective_value(setup.jump_model))
            end
            true_val = x0^2
            @test z_vals[1] <= true_val + 1e-6
            @test z_vals[2] >= true_val - 1e-6
        end
    end

    @testset "Relaxation gap <= 2^(-2L-1)" begin
        for L in [2, 3, 4]
            max_gap = 0.0
            for x0 in range(0.0, 1.0; length = 11)
                z_vals = Float64[]
                for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                    setup = _setup_qa_test(["gen1"], 1:1)
                    JuMP.fix(setup.var_container["gen1", 1], x0; force = true)

                    IOM._add_dnmdt_univariate_approx!(
                        setup.container, MockThermalGen, ["gen1"], 1:1,
                        setup.var_container, 0.0, 1.0, L, DNMDT_META;
                        tighten = false,
                    )
                    expr = IOM.get_expression(
                        setup.container, IOM.QuadraticExpression(),
                        MockThermalGen, DNMDT_META,
                    )

                    JuMP.@objective(setup.jump_model, sense, expr["gen1", 1])
                    JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                    JuMP.set_silent(setup.jump_model)
                    JuMP.optimize!(setup.jump_model)
                    push!(z_vals, JuMP.objective_value(setup.jump_model))
                end
                gap = z_vals[2] - z_vals[1]
                max_gap = max(max_gap, gap)
            end
            # Relaxation gap (UB - LB) bounded by eps_L^2 / 2 = 2^(-2L-1)
            theoretical_bound = 2.0^(-2 * L - 1)
            @test max_gap <= theoretical_bound + 1e-6
        end
    end

    @testset "Constraint structure" begin
        setup = _setup_qa_test(["gen1"], 1:1)
        depth = 3

        IOM._add_dnmdt_univariate_approx!(
            setup.container, MockThermalGen, ["gen1"], 1:1,
            setup.var_container, 0.0, 1.0, depth, DNMDT_META;
            tighten = false,
        )

        # L binary variables for univariate
        n_bin = count(JuMP.is_binary, JuMP.all_variables(setup.jump_model))
        @test n_bin == depth

        # Container keys exist
        @test IOM.has_container_key(
            setup.container, IOM.DNMDTBinaryVariable, MockThermalGen, DNMDT_META,
        )
        @test IOM.has_container_key(
            setup.container, IOM.DNMDTResidualVariable, MockThermalGen, DNMDT_META,
        )
    end

    @testset "Multiple time steps and names" begin
        names = ["gen1", "gen2"]
        ts = 1:3
        setup = _setup_qa_test(names, ts)

        IOM._add_dnmdt_univariate_approx!(
            setup.container, MockThermalGen, names, ts,
            setup.var_container, 0.0, 1.0, 2, DNMDT_META;
            tighten = false,
        )
        expr = IOM.get_expression(
            setup.container, IOM.QuadraticExpression(),
            MockThermalGen, DNMDT_META,
        )

        for name in names, t in ts
            @test expr[name, t] isa JuMP.AffExpr
        end
    end
end

@testset "T-D-NMDT Tightening" begin
    @testset "T-D-NMDT lower bound >= D-NMDT lower bound" begin
        for x0 in [0.15, 0.35, 0.65, 0.85]
            lb_dnmdt = NaN
            lb_tdnmdt = NaN
            for tighten in [false, true]
                setup = _setup_qa_test(["gen1"], 1:1)
                JuMP.fix(setup.var_container["gen1", 1], x0; force = true)

                IOM._add_dnmdt_univariate_approx!(
                    setup.container, MockThermalGen, ["gen1"], 1:1,
                    setup.var_container, 0.0, 1.0, 2, DNMDT_META;
                    tighten = tighten,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.QuadraticExpression(),
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, Min, expr["gen1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL

                if !tighten
                    lb_dnmdt = JuMP.objective_value(setup.jump_model)
                else
                    lb_tdnmdt = JuMP.objective_value(setup.jump_model)
                end
            end
            # T-D-NMDT should be at least as tight
            @test lb_tdnmdt >= lb_dnmdt - 1e-6
            # Both should be valid lower bounds
            @test lb_dnmdt <= x0^2 + 1e-6
            @test lb_tdnmdt <= x0^2 + 1e-6
        end
    end

    @testset "Convergence with depth" begin
        true_val = 0.35^2
        errors = Float64[]
        for L in 1:4
            setup = _setup_qa_test(["gen1"], 1:1)
            JuMP.fix(setup.var_container["gen1", 1], 0.35; force = true)

            IOM._add_dnmdt_univariate_approx!(
                setup.container, MockThermalGen, ["gen1"], 1:1,
                setup.var_container, 0.0, 1.0, L, DNMDT_META;
                tighten = true,
            )
            expr = IOM.get_expression(
                setup.container, IOM.QuadraticExpression(),
                MockThermalGen, DNMDT_META,
            )

            JuMP.@objective(setup.jump_model, Max, expr["gen1", 1])
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)
            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL

            push!(errors, abs(JuMP.objective_value(setup.jump_model) - true_val))
        end
        for i in 2:length(errors)
            @test errors[i] <= errors[i - 1] + 1e-10
        end
    end
end

@testset "D-NMDT Bivariate Approximation" begin
    @testset "Relaxation validity" begin
        test_points = [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9), (0.8, 0.2)]
        for (x0, y0) in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_dnmdt_bilinear_approx!(
                    setup.container, MockThermalGen, ["dev1"], 1:1,
                    setup.x_var_container, setup.y_var_container,
                    0.0, 1.0, 0.0, 1.0, 2, DNMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.BilinearProductExpression(),
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["dev1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                push!(z_vals, JuMP.objective_value(setup.jump_model))
            end
            true_val = x0 * y0
            @test z_vals[1] <= true_val + 1e-6
            @test z_vals[2] >= true_val - 1e-6
        end
    end

    @testset "Relaxation gap <= 2^(-2L-1)" begin
        for L in [2, 3]
            max_gap = 0.0
            for x0 in range(0.05, 0.95; length = 5)
                for y0 in range(0.05, 0.95; length = 5)
                    z_vals = Float64[]
                    for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                        setup = _setup_bilinear_test(["dev1"], 1:1)
                        JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                        JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                        IOM._add_dnmdt_bilinear_approx!(
                            setup.container, MockThermalGen, ["dev1"], 1:1,
                            setup.x_var_container, setup.y_var_container,
                            0.0, 1.0, 0.0, 1.0, L, DNMDT_META,
                        )
                        expr = IOM.get_expression(
                            setup.container, IOM.BilinearProductExpression(),
                            MockThermalGen, DNMDT_META,
                        )

                        JuMP.@objective(setup.jump_model, sense, expr["dev1", 1])
                        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                        JuMP.set_silent(setup.jump_model)
                        JuMP.optimize!(setup.jump_model)
                        push!(z_vals, JuMP.objective_value(setup.jump_model))
                    end
                    gap = z_vals[2] - z_vals[1]
                    max_gap = max(max_gap, gap)
                end
            end
            # Relaxation gap (UB - LB) bounded by eps_L^2 / 2 = 2^(-2L-1)
            theoretical_bound = 2.0^(-2 * L - 1)
            @test max_gap <= theoretical_bound + 1e-6
        end
    end

    @testset "General bounds (non-unit intervals)" begin
        x_min, x_max = 0.2, 0.8
        y_min, y_max = -0.3, 1.5
        test_points = [(0.5, 0.6), (0.3, 1.0), (0.7, -0.1)]
        for (x0, y0) in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_dnmdt_bilinear_approx!(
                    setup.container, MockThermalGen, ["dev1"], 1:1,
                    setup.x_var_container, setup.y_var_container,
                    x_min, x_max, y_min, y_max, 3, DNMDT_META,
                )
                expr = IOM.get_expression(
                    setup.container, IOM.BilinearProductExpression(),
                    MockThermalGen, DNMDT_META,
                )

                JuMP.@objective(setup.jump_model, sense, expr["dev1", 1])
                JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
                JuMP.set_silent(setup.jump_model)
                JuMP.optimize!(setup.jump_model)
                @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
                push!(z_vals, JuMP.objective_value(setup.jump_model))
            end
            true_val = x0 * y0
            @test z_vals[1] <= true_val + 1e-6
            @test z_vals[2] >= true_val - 1e-6
        end
    end

    @testset "Constraint structure" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        depth = 2

        IOM._add_dnmdt_bilinear_approx!(
            setup.container, MockThermalGen, ["dev1"], 1:1,
            setup.x_var_container, setup.y_var_container,
            0.0, 1.0, 0.0, 1.0, depth, DNMDT_META,
        )

        # 2L binary variables for bivariate
        n_bin = count(JuMP.is_binary, JuMP.all_variables(setup.jump_model))
        @test n_bin == 2 * depth

        # Container keys exist
        @test IOM.has_container_key(
            setup.container, IOM.BilinearProductExpression, MockThermalGen,
            DNMDT_META,
        )
    end

    @testset "McCormick toggle" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)

        IOM._add_dnmdt_bilinear_approx!(
            setup.container, MockThermalGen, ["dev1"], 1:1,
            setup.x_var_container, setup.y_var_container,
            0.0, 1.0, 0.0, 1.0, 2, DNMDT_META;
            add_mccormick = false,
        )

        @test !IOM.has_container_key(
            setup.container, IOM.McCormickConstraint, MockThermalGen, DNMDT_META,
        )
    end

    @testset "Fixed-variable correctness" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
        JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

        IOM._add_dnmdt_bilinear_approx!(
            setup.container, MockThermalGen, ["dev1"], 1:1,
            setup.x_var_container, setup.y_var_container,
            0.0, 4.0, 0.0, 4.0, 3, DNMDT_META,
        )
        expr = IOM.get_expression(
            setup.container, IOM.BilinearProductExpression(),
            MockThermalGen, DNMDT_META,
        )

        JuMP.@objective(setup.jump_model, Max, expr["dev1", 1])
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 0.5
    end

    @testset "Multiple time steps" begin
        setup = _setup_bilinear_test(["dev1"], 1:3)

        IOM._add_dnmdt_bilinear_approx!(
            setup.container, MockThermalGen, ["dev1"], 1:3,
            setup.x_var_container, setup.y_var_container,
            0.0, 1.0, 0.0, 1.0, 2, DNMDT_META,
        )
        expr = IOM.get_expression(
            setup.container, IOM.BilinearProductExpression(),
            MockThermalGen, DNMDT_META,
        )

        for t in 1:3
            @test expr["dev1", t] isa JuMP.AffExpr
        end
    end

    @testset "D-NMDT vs HybS comparison" begin
        true_product = 0.4 * 0.7
        for depth in [2, 3]
            # D-NMDT
            setup_d = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup_d.x_var_container["dev1", 1], 0.4; force = true)
            JuMP.fix(setup_d.y_var_container["dev1", 1], 0.7; force = true)

            IOM._add_dnmdt_bilinear_approx!(
                setup_d.container, MockThermalGen, ["dev1"], 1:1,
                setup_d.x_var_container, setup_d.y_var_container,
                0.0, 1.0, 0.0, 1.0, depth, DNMDT_META,
            )
            expr_d = IOM.get_expression(
                setup_d.container, IOM.BilinearProductExpression(),
                MockThermalGen, DNMDT_META,
            )

            JuMP.@objective(setup_d.jump_model, Max, expr_d["dev1", 1])
            JuMP.set_optimizer(setup_d.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup_d.jump_model)
            JuMP.optimize!(setup_d.jump_model)
            dnmdt_gap = abs(JuMP.objective_value(setup_d.jump_model) - true_product)

            # HybS
            setup_h = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup_h.x_var_container["dev1", 1], 0.4; force = true)
            JuMP.fix(setup_h.y_var_container["dev1", 1], 0.7; force = true)

            IOM._add_hybs_bilinear_approx!(
                setup_h.container, MockThermalGen, ["dev1"], 1:1,
                setup_h.x_var_container, setup_h.y_var_container,
                0.0, 1.0, 0.0, 1.0, depth, DNMDT_HYBS_META,
            )
            expr_h = IOM.get_expression(
                setup_h.container, IOM.BilinearProductExpression(),
                MockThermalGen, DNMDT_HYBS_META,
            )

            JuMP.@objective(setup_h.jump_model, Max, expr_h["dev1", 1])
            JuMP.set_optimizer(setup_h.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup_h.jump_model)
            JuMP.optimize!(setup_h.jump_model)
            hybs_gap = abs(JuMP.objective_value(setup_h.jump_model) - true_product)

            # D-NMDT should be at least as tight (same binary budget)
            @test dnmdt_gap <= hybs_gap + 1e-6

            # Both use same number of binaries: 2L
            n_bin_d = count(JuMP.is_binary, JuMP.all_variables(setup_d.jump_model))
            n_bin_h = count(JuMP.is_binary, JuMP.all_variables(setup_h.jump_model))
            @test n_bin_d == n_bin_h
        end
    end
end
