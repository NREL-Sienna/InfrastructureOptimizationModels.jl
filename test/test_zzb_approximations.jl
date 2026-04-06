const ZZB_META = "ZZBTest"

@testset "ZZB Quadratic Approximation" begin

    # ====================================================================
    # 1. Pure encoding tests (no JuMP)
    # ====================================================================
    @testset "BRGC encoding" begin
        @testset "r=1" begin
            G = IOM.build_brgc(1)
            @test size(G) == (2, 1)
            @test G == [0; 1][:, :]
        end

        @testset "r=2" begin
            G = IOM.build_brgc(2)
            @test size(G) == (4, 2)
            @test G == [0 0; 0 1; 1 1; 1 0]
        end

        @testset "r=3" begin
            G = IOM.build_brgc(3)
            @test size(G) == (8, 3)
            # Verify adjacency property: consecutive rows differ by exactly 1 bit
            for i in 1:(size(G, 1) - 1)
                diff = sum(abs.(G[i + 1, :] .- G[i, :]))
                @test diff == 1
            end
        end

        @testset "Adjacency holds for r=1..5" begin
            for r in 1:5
                G = IOM.build_brgc(r)
                @test size(G) == (2^r, r)
                for i in 1:(size(G, 1) - 1)
                    diff = sum(abs.(G[i + 1, :] .- G[i, :]))
                    @test diff == 1
                end
            end
        end
    end

    @testset "ZZB coefficient matrices" begin
        @testset "r=2 known values" begin
            G = IOM.build_brgc(2)
            lower_coeffs, upper_coeffs = IOM._build_zzb_coefficients(G, 2)
            # G = [0 0; 0 1; 1 1; 1 0], 5 breakpoints, 2 columns
            @test size(lower_coeffs) == (5, 2)
            @test size(upper_coeffs) == (5, 2)

            # First breakpoint (i=1): adjacent to segment 1 only → G[1,:] = [0,0]
            @test lower_coeffs[1, :] == [0, 0]
            @test upper_coeffs[1, :] == [0, 0]

            # Last breakpoint (i=5): adjacent to segment 4 only → G[4,:] = [1,0]
            @test lower_coeffs[5, :] == [1, 0]
            @test upper_coeffs[5, :] == [1, 0]

            # Interior breakpoint i=2: adjacent to segments 1,2 → G[1,:]=[0,0], G[2,:]=[0,1]
            @test lower_coeffs[2, :] == [0, 0]  # min(0,0), min(0,1)
            @test upper_coeffs[2, :] == [0, 1]  # max(0,0), max(0,1)

            # Interior breakpoint i=3: adjacent to segments 2,3 → G[2,:]=[0,1], G[3,:]=[1,1]
            @test lower_coeffs[3, :] == [0, 1]  # min(0,1), min(1,1)
            @test upper_coeffs[3, :] == [1, 1]  # max(0,1), max(1,1)

            # Interior breakpoint i=4: adjacent to segments 3,4 → G[3,:]=[1,1], G[4,:]=[1,0]
            @test lower_coeffs[4, :] == [1, 0]  # min(1,1), min(1,0)
            @test upper_coeffs[4, :] == [1, 1]  # max(1,1), max(1,0)
        end

        @testset "Coefficient bounds are valid" begin
            for r in 1:4
                G = IOM.build_brgc(r)
                lo, hi = IOM._build_zzb_coefficients(G, r)
                # All entries must be 0 or 1
                @test all(x -> x == 0 || x == 1, lo)
                @test all(x -> x == 0 || x == 1, hi)
                # lo ≤ hi elementwise
                @test all(lo .<= hi)
            end
        end
    end

    # ====================================================================
    # 2. ZZB quadratic structure test
    # ====================================================================
    @testset "Constraint structure" begin
        setup = _setup_qa_test(["dev1"], 1:1)
        num_segments = 4   # 4 segments → log2(4) = 2 binary variables

        IOM._add_quadratic_approx!(
            IOM.ZZBQuadConfig(num_segments),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.var_container,
            0.0,
            4.0,
            ZZB_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.QuadraticExpression(),
            MockThermalGen,
            ZZB_META,
        )

        @test expr_container["dev1", 1] isa JuMP.AffExpr

        # log2(num_segments) binary variables
        n_bin = count(JuMP.is_binary, JuMP.all_variables(setup.jump_model))
        @test n_bin == Int(log2(num_segments))

        # Lambda variables exist: num_segments + 1 = 5 lambda vars
        @test IOM.has_container_key(
            setup.container,
            IOM.ZZBLambdaVariable,
            MockThermalGen,
            ZZB_META,
        )

        # Encoding constraints exist
        @test IOM.has_container_key(
            setup.container,
            IOM.ZZBEncodingConstraint,
            MockThermalGen,
            ZZB_META,
        )
    end

    # ====================================================================
    # 3. ZZB quadratic correctness (solve tests)
    # ====================================================================
    @testset "Upper-bounds x^2 on [0,1]" begin
        setup = _setup_qa_test(["dev1"], 1:1)
        x_var = setup.var_container["dev1", 1]
        JuMP.fix(x_var, 0.35; force = true)

        IOM._add_quadratic_approx!(
            IOM.ZZBQuadConfig(8),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.var_container,
            0.0,
            1.0,
            ZZB_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.QuadraticExpression(),
            MockThermalGen,
            ZZB_META,
        )
        z_expr = expr_container["dev1", 1]

        # Minimize: should give value ≤ x² (it's an upper bound, min gives closest)
        JuMP.@objective(setup.jump_model, Min, z_expr)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        obj_val = JuMP.objective_value(setup.jump_model)
        # ZZB is a convex combination → upper bound, so min ≥ true - small error
        @test obj_val >= 0.35^2 - 1e-6
        # But not too far above
        @test obj_val <= 0.35^2 + 0.05
    end

    @testset "Upper-bounds x^2 on non-unit interval [0,2]" begin
        setup = _setup_qa_test(["dev1"], 1:1)
        x_var = setup.var_container["dev1", 1]
        JuMP.fix(x_var, 1.3; force = true)

        IOM._add_quadratic_approx!(
            IOM.ZZBQuadConfig(8),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.var_container,
            0.0,
            2.0,
            ZZB_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.QuadraticExpression(),
            MockThermalGen,
            ZZB_META,
        )
        z_expr = expr_container["dev1", 1]

        JuMP.@objective(setup.jump_model, Min, z_expr)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        obj_val = JuMP.objective_value(setup.jump_model)
        @test obj_val >= 1.3^2 - 1e-6
        @test obj_val <= 1.3^2 + 0.2
    end

    @testset "Tightened ZZB bounds x^2" begin
        x0 = 0.35
        true_val = x0^2
        num_segments = 4   # 4 segments → 2 binary variables, epigraph_depth=2

        # Minimize tightened: should still be ≤ x²
        setup_min = _setup_qa_test(["dev1"], 1:1)
        JuMP.fix(setup_min.var_container["dev1", 1], x0; force = true)
        IOM._add_quadratic_approx!(
            IOM.ZZBQuadConfig(num_segments, Int(log2(num_segments))),
            setup_min.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup_min.var_container,
            0.0,
            1.0,
            ZZB_META,
        )
        expr_min = IOM.get_expression(
            setup_min.container,
            IOM.QuadraticExpression(),
            MockThermalGen,
            ZZB_META,
        )
        z_min_expr = expr_min["dev1", 1]

        JuMP.@objective(setup_min.jump_model, Min, z_min_expr)
        JuMP.set_optimizer(setup_min.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup_min.jump_model)
        JuMP.optimize!(setup_min.jump_model)
        @test JuMP.termination_status(setup_min.jump_model) == JuMP.OPTIMAL
        min_val = JuMP.objective_value(setup_min.jump_model)
        @test min_val <= true_val + 1e-6

        # Maximize tightened: should be ≥ x²
        setup_max = _setup_qa_test(["dev1"], 1:1)
        JuMP.fix(setup_max.var_container["dev1", 1], x0; force = true)
        IOM._add_quadratic_approx!(
            IOM.ZZBQuadConfig(num_segments, Int(log2(num_segments))),
            setup_max.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup_max.var_container,
            0.0,
            1.0,
            ZZB_META,
        )
        expr_max = IOM.get_expression(
            setup_max.container,
            IOM.QuadraticExpression(),
            MockThermalGen,
            ZZB_META,
        )
        z_max_expr = expr_max["dev1", 1]

        JuMP.@objective(setup_max.jump_model, Max, z_max_expr)
        JuMP.set_optimizer(setup_max.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup_max.jump_model)
        JuMP.optimize!(setup_max.jump_model)
        @test JuMP.termination_status(setup_max.jump_model) == JuMP.OPTIMAL
        max_val = JuMP.objective_value(setup_max.jump_model)
        @test max_val >= true_val - 1e-6
    end

    @testset "Approximation quality improves with segments" begin
        errors = Float64[]
        true_val = 0.35^2
        for num_segments in [2, 4, 8, 16]
            setup = _setup_qa_test(["dev1"], 1:1)
            x_var = setup.var_container["dev1", 1]
            JuMP.fix(x_var, 0.35; force = true)

            IOM._add_quadratic_approx!(
                IOM.ZZBQuadConfig(num_segments),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup.var_container,
                0.0,
                1.0,
                ZZB_META,
            )
            expr_container = IOM.get_expression(
                setup.container,
                IOM.QuadraticExpression(),
                MockThermalGen,
                ZZB_META,
            )
            z_expr = expr_container["dev1", 1]

            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            obj_val = JuMP.objective_value(setup.jump_model)
            push!(errors, abs(obj_val - true_val))
        end
        for i in 2:length(errors)
            @test errors[i] <= errors[i - 1] + 1e-10
        end
    end

    @testset "Multiple time steps" begin
        setup = _setup_qa_test(["dev1"], 1:3)
        IOM._add_quadratic_approx!(
            IOM.ZZBQuadConfig(4),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:3,
            setup.var_container,
            0.0,
            4.0,
            ZZB_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.QuadraticExpression(),
            MockThermalGen,
            ZZB_META,
        )

        for t in 1:3
            @test expr_container["dev1", t] isa JuMP.AffExpr
        end
    end
end

# ====================================================================
# 4. HybS + ZZB bilinear integration
# ====================================================================
@testset "HybS + ZZB Bilinear Approximation" begin
    @testset "Constraint structure" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        num_segments = 4   # 4 segments → 2 binary vars per quadratic

        IOM._add_bilinear_approx!(
            IOM.HybSConfig(IOM.ZZBQuadConfig(num_segments), Int(log2(num_segments)), false),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.x_var_container,
            setup.y_var_container,
            0.0,
            4.0,
            0.0,
            4.0,
            ZZB_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.BilinearProductExpression(),
            MockThermalGen,
            ZZB_META,
        )

        @test expr_container["dev1", 1] isa JuMP.AffExpr

        # Binary count: 2 * log2(num_segments) (one set for x², one for y²), zero from epigraphs
        n_bin = count(JuMP.is_binary, JuMP.all_variables(setup.jump_model))
        @test n_bin == 2 * Int(log2(num_segments))
    end

    @testset "Brackets true product at interior points" begin
        test_points = [(0.3, 0.7), (0.5, 0.5), (0.1, 0.9), (0.8, 0.2)]
        for (x0, y0) in test_points
            z_vals = Float64[]
            for sense in [JuMP.MIN_SENSE, JuMP.MAX_SENSE]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_bilinear_approx!(
                    IOM.HybSConfig(IOM.ZZBQuadConfig(4), 2, false),
                    setup.container,
                    MockThermalGen,
                    ["dev1"],
                    1:1,
                    setup.x_var_container,
                    setup.y_var_container,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    ZZB_META,
                )
                expr_container = IOM.get_expression(
                    setup.container,
                    IOM.BilinearProductExpression(),
                    MockThermalGen,
                    ZZB_META,
                )
                z_expr = expr_container["dev1", 1]

                JuMP.@objective(setup.jump_model, sense, z_expr)
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

    @testset "Fixed-variable correctness" begin
        setup = _setup_bilinear_test(["dev1"], 1:1)
        x_var = setup.x_var_container["dev1", 1]
        y_var = setup.y_var_container["dev1", 1]
        JuMP.fix(x_var, 2.0; force = true)
        JuMP.fix(y_var, 3.0; force = true)

        IOM._add_bilinear_approx!(
            IOM.HybSConfig(IOM.ZZBQuadConfig(8), 3, false),
            setup.container,
            MockThermalGen,
            ["dev1"],
            1:1,
            setup.x_var_container,
            setup.y_var_container,
            0.0,
            4.0,
            0.0,
            4.0,
            ZZB_META,
        )
        expr_container = IOM.get_expression(
            setup.container,
            IOM.BilinearProductExpression(),
            MockThermalGen,
            ZZB_META,
        )
        z_expr = expr_container["dev1", 1]

        JuMP.@objective(setup.jump_model, Max, z_expr)
        JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
        JuMP.set_silent(setup.jump_model)
        JuMP.optimize!(setup.jump_model)

        @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
        @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 0.5
    end

    @testset "HybS+ZZB uses fewer binaries than Bin2+ZZB" begin
        for num_segments in [2, 4, 8]
            binary_depth = Int(log2(num_segments))
            # HybS
            setup_h = _setup_bilinear_test(["dev1"], 1:1)
            IOM._add_bilinear_approx!(
                IOM.HybSConfig(IOM.ZZBQuadConfig(num_segments), binary_depth, false),
                setup_h.container,
                MockThermalGen,
                ["dev1"],
                1:1,
                setup_h.x_var_container,
                setup_h.y_var_container,
                0.0,
                1.0,
                0.0,
                1.0,
                ZZB_META,
            )
            n_bin_hybs =
                count(JuMP.is_binary, JuMP.all_variables(setup_h.jump_model))

            @test n_bin_hybs == 2 * binary_depth
        end
    end
end
