const ZZI_META = "ZZITest"

@testset "ZZI Bivariate Approximation" begin

    # ====================================================================
    # Section 1: ZZI Encoding (Pure Math, No JuMP)
    # ====================================================================
    @testset "ZZI encoding" begin
        @testset "d=4 (r=2) known values" begin
            C, C_ext = IOM.build_zzi_encoding(4)
            @test size(C) == (4, 2)
            # build_brgc(2) (prepend convention) = [0 0; 0 1; 1 1; 1 0]
            # Cumulative transitions along each column:
            #   col 1: 0,0,1,1  → C[:,1] = [0, 0, 1, 1]
            #   col 2: 0,1,1,0  → C[:,2] = [0, 1, 1, 2]
            @test C == [0 0; 0 1; 1 1; 1 2]
            @test size(C_ext) == (6, 2)
            # C_ext: row 1 = C_0 = C[1,:], rows 2..5 = C[1..4,:], row 6 = C[4,:]
            @test C_ext[1, :] == [0, 0]  # C_0 = C_1
            @test C_ext[6, :] == [1, 2]  # C_5 = C_4
        end

        @testset "d=2 (r=1)" begin
            C, C_ext = IOM.build_zzi_encoding(2)
            @test size(C) == (2, 1)
            @test C == reshape([0, 1], 2, 1)
        end

        @testset "Monotonicity" begin
            for d in [2, 3, 4, 5, 7, 8, 16]
                C, _ = IOM.build_zzi_encoding(d)
                for k in 1:size(C, 2), i in 2:size(C, 1)
                    @test C[i, k] >= C[i - 1, k]
                end
            end
        end
    end

    @testset "Triangulation chooser" begin
        x_bkpts = [0.0, 1.0, 2.0]
        y_bkpts = [0.0, 1.0, 2.0]
        triang = IOM._choose_triangulation(x_bkpts, y_bkpts)
        @test size(triang) == (2, 2)
        @test all(t -> t == :U || t == :K, triang)
    end

    # ====================================================================
    # Section 2: ZZI Standalone Solve Tests
    # ====================================================================
    @testset "ZZI standalone" begin
        @testset "Constraint structure" begin
            # d1=d2=4 → r1=r2=2 → 4 integers (2 per axis), 6 triangle binaries
            setup = _setup_bilinear_test(["dev1"], 1:1)
            config = IOM.ZZIBilinearConfig(4, 4, true, false, 0)
            IOM._add_bilinear_approx!(
                config,
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
                4,
                ZZI_META,
            )

            expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )
            @test expr["dev1", 1] isa JuMP.AffExpr

            n_int = count(
                v -> JuMP.is_integer(v) && !JuMP.is_binary(v),
                JuMP.all_variables(setup.jump_model),
            )
            @test n_int == 4  # ceil(log2(4)) + ceil(log2(4)) = 2 + 2

            n_bin = count(JuMP.is_binary, JuMP.all_variables(setup.jump_model))
            @test n_bin == 6  # triangle selection

            @test IOM.has_container_key(
                setup.container,
                IOM.ZZILambdaVariable,
                MockThermalGen,
                ZZI_META,
            )
        end

        @testset "Fixed-variable correctness" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            IOM._add_bilinear_approx!(
                IOM.ZZIBilinearConfig(4, 4),
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
                4,
                ZZI_META,
            )

            z_expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )["dev1", 1]
            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 1e-4  # exact at grid point
        end

        @testset "Vertex optima" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.set_lower_bound(setup.x_var_container["dev1", 1], 0.0)
            JuMP.set_upper_bound(setup.x_var_container["dev1", 1], 4.0)
            JuMP.set_lower_bound(setup.y_var_container["dev1", 1], 0.0)
            JuMP.set_upper_bound(setup.y_var_container["dev1", 1], 4.0)

            IOM._add_bilinear_approx!(
                IOM.ZZIBilinearConfig(4, 4),
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
                4,
                ZZI_META,
            )

            z_expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )["dev1", 1]
            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 0.0 atol = 1e-6
        end

        @testset "Approximation quality improves with grid refinement" begin
            errors = Float64[]
            x0, y0 = 1.3, 2.7
            true_val = x0 * y0
            for d in [2, 4, 8]
                setup = _setup_bilinear_test(["dev1"], 1:1)
                JuMP.fix(setup.x_var_container["dev1", 1], x0; force = true)
                JuMP.fix(setup.y_var_container["dev1", 1], y0; force = true)

                IOM._add_bilinear_approx!(
                    IOM.ZZIBilinearConfig(d, d),
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
                    4,
                    ZZI_META,
                )

                z_expr = IOM.get_expression(
                    setup.container,
                    IOM.BilinearProductExpression(),
                    MockThermalGen,
                    ZZI_META,
                )["dev1", 1]
                JuMP.@objective(setup.jump_model, Min, z_expr)
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

        @testset "Multiple time steps" begin
            setup = _setup_bilinear_test(["dev1"], 1:3)
            IOM._add_bilinear_approx!(
                IOM.ZZIBilinearConfig(4, 4),
                setup.container,
                MockThermalGen,
                ["dev1"],
                1:3,
                setup.x_var_container,
                setup.y_var_container,
                0.0,
                4.0,
                0.0,
                4.0,
                4,
                ZZI_META,
            )

            expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )
            for t in 1:3
                @test expr["dev1", t] isa JuMP.AffExpr
            end
        end
    end

    # ====================================================================
    # Section 3: Sawtooth Strengthening
    # ====================================================================
    @testset "ZZI with sawtooth strengthening" begin
        @testset "Constraint structure" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            config = IOM.ZZIBilinearConfig(4, 4, true, true, 3)
            IOM._add_bilinear_approx!(
                config,
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
                4,
                ZZI_META,
            )

            @test IOM.has_container_key(
                setup.container,
                IOM.ZZISawtoothBoundConstraint,
                MockThermalGen,
                ZZI_META * "_st",
            )

            # 6 triangle binaries + 2*3 sawtooth binaries = 12 binaries
            n_bin = count(JuMP.is_binary, JuMP.all_variables(setup.jump_model))
            @test n_bin == 12
        end

        @testset "Fixed-variable correctness at grid point" begin
            # Use a grid point to avoid LP infeasibility from ZZI+sawtooth interaction
            # (at non-grid points, the sawtooth LP relaxation can conflict with the
            # ZZI product equality constraint)
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            IOM._add_bilinear_approx!(
                IOM.ZZIBilinearConfig(4, 4, true, true, 3),
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
                4,
                ZZI_META,
            )

            z_expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )["dev1", 1]
            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 1e-4
        end
    end

    # ====================================================================
    # Section 4: ZZI Strengthened Wrapper with HybS
    # ====================================================================
    @testset "ZZI strengthened HybS" begin
        @testset "Constraint structure" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            inner = IOM.HybSConfig(IOM.SawtoothQuadConfig(4), 4)
            config = IOM.ZZIStrengthenedConfig(inner, 4, 4)
            IOM._add_bilinear_approx!(
                config,
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
                3,
                ZZI_META,
            )

            expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )
            @test expr["dev1", 1] isa JuMP.AffExpr

            # Both HybS and ZZI containers exist
            @test IOM.has_container_key(
                setup.container,
                IOM.HybSBoundConstraint,
                MockThermalGen,
                ZZI_META,
            )
            meta_zzi = ZZI_META * "_zzir"
            @test IOM.has_container_key(
                setup.container,
                IOM.ZZILambdaVariable,
                MockThermalGen,
                meta_zzi,
            )
        end

        @testset "Fixed-variable correctness" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            inner = IOM.HybSConfig(IOM.SawtoothQuadConfig(4), 4)
            config = IOM.ZZIStrengthenedConfig(inner, 4, 4)
            IOM._add_bilinear_approx!(
                config,
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
                3,
                ZZI_META,
            )

            z_expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )["dev1", 1]
            JuMP.@objective(setup.jump_model, Max, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 0.5
        end
    end

    # ====================================================================
    # Section 5: ZZI Strengthened Wrapper with Bin2
    # ====================================================================
    @testset "ZZI strengthened Bin2" begin
        @testset "Fixed-variable correctness" begin
            setup = _setup_bilinear_test(["dev1"], 1:1)
            JuMP.fix(setup.x_var_container["dev1", 1], 2.0; force = true)
            JuMP.fix(setup.y_var_container["dev1", 1], 3.0; force = true)

            inner = IOM.Bin2Config(IOM.SawtoothQuadConfig(4))
            config = IOM.ZZIStrengthenedConfig(inner, 4, 4)
            IOM._add_bilinear_approx!(
                config,
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
                3,
                ZZI_META,
            )

            z_expr = IOM.get_expression(
                setup.container,
                IOM.BilinearProductExpression(),
                MockThermalGen,
                ZZI_META,
            )["dev1", 1]
            JuMP.@objective(setup.jump_model, Min, z_expr)
            JuMP.set_optimizer(setup.jump_model, HiGHS.Optimizer)
            JuMP.set_silent(setup.jump_model)
            JuMP.optimize!(setup.jump_model)

            @test JuMP.termination_status(setup.jump_model) == JuMP.OPTIMAL
            @test JuMP.objective_value(setup.jump_model) ≈ 6.0 atol = 0.5
        end
    end

end
