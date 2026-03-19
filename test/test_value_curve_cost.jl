"""
Unit tests for delta PWL offer curve objective formulation.

Tests functions in src/objective_function/value_curve_cost.jl and
src/objective_function/objective_function_pwl_delta.jl, focusing on
structural correctness (variable/constraint counts via moi_tests) and
objective coefficient accuracy.

Complements test_ts_value_curve_objective.jl, which covers the
add_variable_cost_to_objective! dispatch for
CostCurve{TimeSeriesPiecewiseIncrementalCurve}.
"""

function moi_tests(
    container::IOM.OptimizationContainer,
    vars::Int,
    interval::Int,
    lessthan::Int,
    greaterthan::Int,
    equalto::Int,
    binary::Bool,
    lessthan_quadratic::Union{Int, Nothing} = nothing,
)
    jump_model = IOM.get_jump_model(container)
    @test JuMP.num_variables(jump_model) == vars
    @test JuMP.num_constraints(jump_model, IOM.GAE, MOI.Interval{Float64}) == interval
    @test JuMP.num_constraints(jump_model, IOM.GAE, MOI.LessThan{Float64}) == lessthan
    @test JuMP.num_constraints(jump_model, IOM.GAE, MOI.GreaterThan{Float64}) == greaterthan
    @test JuMP.num_constraints(jump_model, IOM.GAE, MOI.EqualTo{Float64}) == equalto
    @test ((JuMP.VariableRef, MOI.ZeroOne) in JuMP.list_of_constraint_types(jump_model)) ==
          binary
    !isnothing(lessthan_quadratic) &&
        @test JuMP.num_constraints(
            jump_model,
            JuMP.GenericQuadExpr{Float64, VariableRef},
            MOI.LessThan{Float64},
        ) ==
              lessthan_quadratic
    return
end

"""
Helper: set up a container with a power variable for delta PWL offer tests.
Uses TestVariableType for the power variable.
"""
function setup_offer_pwl_container(
    time_steps::UnitRange{Int},
    names::Vector{String};
    base_power::Float64 = 100.0,
    resolution::Dates.Period = Dates.Hour(1),
)
    sys = MockSystem(base_power)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = resolution,
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), MockDeterministic)
    IOM.set_time_steps!(container, time_steps)
    var_container = IOM.add_variable_container!(
        container, TestVariableType(), MockThermalGen, names, time_steps)
    jump_model = IOM.get_jump_model(container)
    for name in names, t in time_steps
        var_container[name, t] =
            JuMP.@variable(jump_model, base_name = "p_$(name)_$t")
    end
    return container
end

@testset "Value Curve Cost — Delta PWL Formulation" begin

    # =========================================================================
    # get_pwl_cost_expression
    # =========================================================================
    @testset "get_pwl_cost_expression" begin
        @testset "returns AffExpr with correct coefficients" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            vars = [JuMP.@variable(jump_model) for _ in 1:3]
            slopes = [10.0, 20.0, 30.0]
            expr = IOM.get_pwl_cost_expression(vars, slopes, 1.0)
            @test expr isa JuMP.AffExpr
            for (i, v) in enumerate(vars)
                @test JuMP.coefficient(expr, v) ≈ slopes[i] atol = 1e-10
            end
        end

        @testset "scales coefficients by dt multiplier" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            vars = [JuMP.@variable(jump_model), JuMP.@variable(jump_model)]
            slopes = [5.0, 15.0]
            dt = 0.25  # 15-min resolution
            expr = IOM.get_pwl_cost_expression(vars, slopes, dt)
            @test JuMP.coefficient(expr, vars[1]) ≈ 5.0 * dt atol = 1e-10
            @test JuMP.coefficient(expr, vars[2]) ≈ 15.0 * dt atol = 1e-10
        end

        @testset "negative multiplier (decremental direction)" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            vars = [JuMP.@variable(jump_model), JuMP.@variable(jump_model)]
            slopes = [8.0, 16.0]
            expr =
                IOM.get_pwl_cost_expression(vars, slopes, IOM.OBJECTIVE_FUNCTION_NEGATIVE)
            @test JuMP.coefficient(expr, vars[1]) ≈ -8.0 atol = 1e-10
            @test JuMP.coefficient(expr, vars[2]) ≈ -16.0 atol = 1e-10
        end

        @testset "single segment" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            v = JuMP.@variable(jump_model)
            expr = IOM.get_pwl_cost_expression([v], [42.0], 1.0)
            @test JuMP.coefficient(expr, v) ≈ 42.0 atol = 1e-10
        end

        @testset "zero slope produces zero coefficient" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            vars = [JuMP.@variable(jump_model), JuMP.@variable(jump_model)]
            expr = IOM.get_pwl_cost_expression(vars, [0.0, 5.0], 1.0)
            @test JuMP.coefficient(expr, vars[1]) ≈ 0.0 atol = 1e-10
            @test JuMP.coefficient(expr, vars[2]) ≈ 5.0 atol = 1e-10
        end
    end

    # =========================================================================
    # add_pwl_variables!
    # =========================================================================
    @testset "add_pwl_variables!" begin
        @testset "Inf upper_bound: lower bound 0, no upper bound" begin
            container = make_test_container(1:1)
            pwl_vars = IOM.add_pwl_variables!(
                container,
                IOM.PiecewiseLinearBlockIncrementalOffer,
                MockThermalGen,
                "gen1",
                1,
                3;
                upper_bound = Inf,
            )
            @test length(pwl_vars) == 3
            for v in pwl_vars
                @test JuMP.lower_bound(v) == 0.0
                @test !JuMP.has_upper_bound(v)
            end
        end

        @testset "finite upper_bound is applied" begin
            container = make_test_container(1:1)
            ub = 0.5
            pwl_vars = IOM.add_pwl_variables!(
                container,
                IOM.PiecewiseLinearBlockIncrementalOffer,
                MockThermalGen,
                "gen1",
                1,
                2;
                upper_bound = ub,
            )
            for v in pwl_vars
                @test JuMP.lower_bound(v) == 0.0
                @test JuMP.upper_bound(v) ≈ ub atol = 1e-10
            end
        end

        @testset "incremental and decremental use separate containers" begin
            container = make_test_container(1:1)
            IOM.add_pwl_variables!(
                container, IOM.PiecewiseLinearBlockIncrementalOffer,
                MockThermalGen, "gen1", 1, 2; upper_bound = Inf)
            IOM.add_pwl_variables!(
                container, IOM.PiecewiseLinearBlockDecrementalOffer,
                MockThermalGen, "gen1", 1, 2; upper_bound = Inf)
            @test IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
            @test IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockDecrementalOffer, MockThermalGen)
        end

        @testset "variables are retrievable from sparse container" begin
            container = make_test_container(1:2)
            name = "gen1"
            n_segs = 3
            for t in 1:2
                IOM.add_pwl_variables!(
                    container, IOM.PiecewiseLinearBlockIncrementalOffer,
                    MockThermalGen, name, t, n_segs; upper_bound = Inf)
            end
            delta_container = IOM.get_variable(
                container, IOM.PiecewiseLinearBlockIncrementalOffer(), MockThermalGen)
            for t in 1:2, k in 1:n_segs
                @test delta_container[(name, k, t)] isa JuMP.VariableRef
            end
        end
    end

    # =========================================================================
    # add_pwl_block_offer_constraints!
    # =========================================================================
    @testset "add_pwl_block_offer_constraints!" begin
        @testset "linking EqualTo and block LessThan constraints created" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            # 2 delta vars, 3 breakpoints
            power_var = JuMP.@variable(jump_model, base_name = "p")
            pwl_vars = [JuMP.@variable(jump_model), JuMP.@variable(jump_model)]
            breakpoints = [0.0, 0.5, 1.0]

            # Use a JuMP DenseAxisArray as a stand-in for the constraint container
            con_container = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
                undef, ["gen1"], 1:1)
            IOM.add_pwl_block_offer_constraints!(
                jump_model, con_container, "gen1", 1,
                power_var, pwl_vars, breakpoints)

            @test JuMP.num_constraints(jump_model, IOM.GAE, MOI.EqualTo{Float64}) == 1
            @test JuMP.num_constraints(jump_model, IOM.GAE, MOI.LessThan{Float64}) == 2
        end

        @testset "linking constraint rhs is zero (standard form)" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            power_var = JuMP.@variable(jump_model, base_name = "p")
            pwl_vars = [JuMP.@variable(jump_model), JuMP.@variable(jump_model)]
            breakpoints = [0.0, 0.5, 1.0]

            con_container = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
                undef, ["gen1"], 1:1)
            IOM.add_pwl_block_offer_constraints!(
                jump_model, con_container, "gen1", 1,
                power_var, pwl_vars, breakpoints)

            linking_con = con_container["gen1", 1]
            con_obj = JuMP.constraint_object(linking_con)
            @test con_obj.set.value ≈ 0.0 atol = 1e-10
            # power_var has coefficient +1 in LHS
            @test JuMP.coefficient(con_obj.func, power_var) ≈ 1.0 atol = 1e-10
            # delta vars have coefficient -1 in LHS
            for v in pwl_vars
                @test JuMP.coefficient(con_obj.func, v) ≈ -1.0 atol = 1e-10
            end
        end

        @testset "block width upper bounds match breakpoint widths" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            power_var = JuMP.@variable(jump_model, base_name = "p")
            pwl_vars = [JuMP.@variable(jump_model, base_name = "d1"),
                JuMP.@variable(jump_model, base_name = "d2")]
            breakpoints = [0.0, 0.3, 1.0]

            con_container = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
                undef, ["gen1"], 1:1)
            IOM.add_pwl_block_offer_constraints!(
                jump_model, con_container, "gen1", 1,
                power_var, pwl_vars, breakpoints)

            # Width constraints are stored as upper bounds on the variables
            # Block 1: delta_1 <= 0.3 - 0.0 = 0.3
            # Block 2: delta_2 <= 1.0 - 0.3 = 0.7
            lessthan_cons = JuMP.all_constraints(jump_model, IOM.GAE, MOI.LessThan{Float64})
            rhs_values = sort([
                JuMP.constraint_object(c).set.upper for c in lessthan_cons])
            @test rhs_values[1] ≈ 0.3 atol = 1e-10
            @test rhs_values[2] ≈ 0.7 atol = 1e-10
        end

        @testset "min_power_offset shifts linking constraint RHS" begin
            container = make_test_container(1:1)
            jump_model = IOM.get_jump_model(container)
            power_var = JuMP.@variable(jump_model, base_name = "p")
            pwl_vars = [JuMP.@variable(jump_model)]
            breakpoints = [0.3, 1.0]
            offset = 0.3  # P_min in p.u.

            con_container = JuMP.Containers.DenseAxisArray{JuMP.ConstraintRef}(
                undef, ["gen1"], 1:1)
            IOM.add_pwl_block_offer_constraints!(
                jump_model, con_container, "gen1", 1,
                power_var, pwl_vars, breakpoints, offset)

            linking_con = con_container["gen1", 1]
            con_obj = JuMP.constraint_object(linking_con)
            # power_var == delta + offset
            # JuMP normalizes constants to RHS: (power_var - delta) in EqualTo(offset)
            @test con_obj.set.value ≈ offset atol = 1e-10
        end
    end

    # =========================================================================
    # _add_ts_incremental_pwl_cost! — structure via moi_tests
    # =========================================================================
    @testset "_add_ts_incremental_pwl_cost! — structure (moi_tests)" begin
        @testset "1 device, 3 time steps, 2 segments — IncrementalOffer" begin
            # vars:     3 power + 3*2 delta = 9
            # lessthan: 3*2 block-width upper bounds = 6
            # equalto:  3 linking constraints = 3
            time_steps = 1:3
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 3)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 3)
            for i in 1:1, t in 1:3
                slopes_mat[i, t] = [10.0, 20.0]
                bp_mat[i, t] = [0.0, 50.0, 100.0]
            end
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.IncrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            moi_tests(container, 9, 0, 6, 0, 3, false)
        end

        @testset "1 device, 2 time steps, 2 segments — DecrementalOffer" begin
            # vars:     2 power + 2*2 delta = 6
            # lessthan: 2*2 = 4
            # equalto:  2
            time_steps = 1:2
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            for i in 1:1, t in 1:2
                slopes_mat[i, t] = [8.0, 16.0]
                bp_mat[i, t] = [0.0, 40.0, 100.0]
            end
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps;
                dir = IOM.DecrementalOffer())

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.DecrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            moi_tests(container, 6, 0, 4, 0, 2, false)
        end

        @testset "2 devices, 2 time steps, 3 segments — IncrementalOffer" begin
            # vars:     4 power + 2*2*3 delta = 16
            # lessthan: 2*2*3 = 12
            # equalto:  2*2 = 4
            time_steps = 1:2
            names = ["gen1", "gen2"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 2, 2)
            bp_mat = Matrix{Vector{Float64}}(undef, 2, 2)
            for i in 1:2, t in 1:2
                slopes_mat[i, t] = [5.0, 10.0, 15.0]
                bp_mat[i, t] = [0.0, 30.0, 70.0, 100.0]
            end
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            for name in names
                device = make_mock_thermal(name)
                IOM._add_ts_incremental_pwl_cost!(
                    IOM.IncrementalOffer(), container, device,
                    TestVariableType(), TestFormulation())
            end

            moi_tests(container, 16, 0, 12, 0, 4, false)
        end

        @testset "1 device, 4 time steps, 1 segment — minimal structure" begin
            # vars:     4 power + 4*1 delta = 8
            # lessthan: 4*1 = 4
            # equalto:  4
            time_steps = 1:4
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 4)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 4)
            for i in 1:1, t in 1:4
                slopes_mat[i, t] = [25.0]
                bp_mat[i, t] = [0.0, 100.0]
            end
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.IncrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            moi_tests(container, 8, 0, 4, 0, 4, false)
        end
    end

    # =========================================================================
    # _add_ts_incremental_pwl_cost! — objective coefficient verification
    # =========================================================================
    @testset "_add_ts_incremental_pwl_cost! — objective coefficients" begin
        @testset "incremental costs routed to variant expression" begin
            time_steps = 1:2
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            # slopes stored as raw $/MWh; after _fill (×base_power=100): [1000, 2000]
            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            for i in 1:1, t in 1:2
                slopes_mat[i, t] = [10.0, 20.0]
                bp_mat[i, t] = [0.0, 50.0, 100.0]
            end
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.IncrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            # Invariant expression must be empty (time-series path always uses variant)
            @test count_objective_terms(container; variant = false) == 0
            # Variant: 2 time steps × 2 segments = 4 terms
            @test count_objective_terms(container; variant = true) == 4

            delta_container = IOM.get_variable(
                container, IOM.PiecewiseLinearBlockIncrementalOffer(), MockThermalGen)
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)

            # dt = 1.0 (hourly); coefficient = slope * base_power * dt
            for t in time_steps
                @test JuMP.coefficient(variant, delta_container[("gen1", 1, t)]) ≈
                      10.0 * 100.0 * 1.0 atol = 1e-10
                @test JuMP.coefficient(variant, delta_container[("gen1", 2, t)]) ≈
                      20.0 * 100.0 * 1.0 atol = 1e-10
            end
        end

        @testset "decremental costs have negative objective sign" begin
            time_steps = 1:1
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            slopes_mat[1, 1] = [10.0, 20.0]
            bp_mat[1, 1] = [0.0, 50.0, 100.0]
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps;
                dir = IOM.DecrementalOffer())

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.DecrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            delta_container = IOM.get_variable(
                container, IOM.PiecewiseLinearBlockDecrementalOffer(), MockThermalGen)
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)

            # sign = OBJECTIVE_FUNCTION_NEGATIVE = -1.0
            @test JuMP.coefficient(variant, delta_container[("gen1", 1, 1)]) ≈
                  -10.0 * 100.0 atol = 1e-10
            @test JuMP.coefficient(variant, delta_container[("gen1", 2, 1)]) ≈
                  -20.0 * 100.0 atol = 1e-10
        end

        @testset "15-min resolution scales coefficients by dt = 0.25" begin
            time_steps = 1:2
            names = ["gen1"]
            container = setup_offer_pwl_container(
                time_steps, names; resolution = Dates.Minute(15))

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            for i in 1:1, t in 1:2
                slopes_mat[i, t] = [10.0, 20.0]
                bp_mat[i, t] = [0.0, 50.0, 100.0]
            end
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.IncrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            delta_container = IOM.get_variable(
                container, IOM.PiecewiseLinearBlockIncrementalOffer(), MockThermalGen)
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)

            dt = 15.0 / 60.0  # = 0.25
            for t in time_steps
                @test JuMP.coefficient(variant, delta_container[("gen1", 1, t)]) ≈
                      10.0 * 100.0 * dt atol = 1e-10
                @test JuMP.coefficient(variant, delta_container[("gen1", 2, t)]) ≈
                      20.0 * 100.0 * dt atol = 1e-10
            end
        end

        @testset "time-varying slopes produce distinct per-timestep coefficients" begin
            time_steps = 1:2
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 2)
            slopes_mat[1, 1] = [10.0, 20.0]   # t=1
            slopes_mat[1, 2] = [15.0, 25.0]   # t=2
            for t in 1:2
                bp_mat[1, t] = [0.0, 50.0, 100.0]
            end
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.IncrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            delta_container = IOM.get_variable(
                container, IOM.PiecewiseLinearBlockIncrementalOffer(), MockThermalGen)
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)

            # t=1: [10, 20] * 100 * 1.0
            @test JuMP.coefficient(variant, delta_container[("gen1", 1, 1)]) ≈ 1000.0 atol =
                1e-10
            @test JuMP.coefficient(variant, delta_container[("gen1", 2, 1)]) ≈ 2000.0 atol =
                1e-10
            # t=2: [15, 25] * 100 * 1.0
            @test JuMP.coefficient(variant, delta_container[("gen1", 1, 2)]) ≈ 1500.0 atol =
                1e-10
            @test JuMP.coefficient(variant, delta_container[("gen1", 2, 2)]) ≈ 2500.0 atol =
                1e-10
        end

        @testset "non-default base_power correctly scales coefficients" begin
            time_steps = 1:1
            names = ["gen1"]
            base_power = 200.0
            container =
                setup_offer_pwl_container(time_steps, names; base_power = base_power)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            # stored slopes [0.05, 0.1]; after fill (×200): [10, 20]
            slopes_mat[1, 1] = [0.05, 0.1]
            bp_mat[1, 1] = [0.0, 50.0, 100.0]
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            device = make_mock_thermal("gen1"; base_power = base_power)
            IOM._add_ts_incremental_pwl_cost!(
                IOM.IncrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            delta_container = IOM.get_variable(
                container, IOM.PiecewiseLinearBlockIncrementalOffer(), MockThermalGen)
            obj = IOM.get_objective_expression(container)
            variant = IOM.get_variant_terms(obj)

            # 0.05 * 200 * 1.0 = 10.0 and 0.1 * 200 * 1.0 = 20.0
            @test JuMP.coefficient(variant, delta_container[("gen1", 1, 1)]) ≈ 10.0 atol =
                1e-10
            @test JuMP.coefficient(variant, delta_container[("gen1", 2, 1)]) ≈ 20.0 atol =
                1e-10
        end
    end

    # =========================================================================
    # Container key presence checks
    # =========================================================================
    @testset "_add_ts_incremental_pwl_cost! — container keys" begin
        @testset "incremental: creates incremental variable and constraint keys" begin
            time_steps = 1:1
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            slopes_mat[1, 1] = [10.0, 20.0]
            bp_mat[1, 1] = [0.0, 50.0, 100.0]
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps)

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.IncrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            @test IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
            @test IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
                MockThermalGen)
            # Decremental keys must NOT be created
            @test !IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockDecrementalOffer, MockThermalGen)
        end

        @testset "decremental: creates decremental variable and constraint keys" begin
            time_steps = 1:1
            names = ["gen1"]
            container = setup_offer_pwl_container(time_steps, names)

            slopes_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            bp_mat = Matrix{Vector{Float64}}(undef, 1, 1)
            slopes_mat[1, 1] = [10.0, 20.0]
            bp_mat[1, 1] = [0.0, 50.0, 100.0]
            setup_delta_pwl_parameters!(
                container, MockThermalGen, names, slopes_mat, bp_mat, time_steps;
                dir = IOM.DecrementalOffer())

            device = make_mock_thermal("gen1")
            IOM._add_ts_incremental_pwl_cost!(
                IOM.DecrementalOffer(), container, device,
                TestVariableType(), TestFormulation())

            @test IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockDecrementalOffer, MockThermalGen)
            @test IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockDecrementalOfferConstraint,
                MockThermalGen)
            @test !IOM.has_container_key(
                container, IOM.PiecewiseLinearBlockIncrementalOffer, MockThermalGen)
        end
    end

    # =========================================================================
    # OfferDirection dispatch table
    # =========================================================================
    @testset "OfferDirection dispatch helpers" begin
        @test IOM._block_offer_var(IOM.IncrementalOffer()) ==
              IOM.PiecewiseLinearBlockIncrementalOffer
        @test IOM._block_offer_var(IOM.DecrementalOffer()) ==
              IOM.PiecewiseLinearBlockDecrementalOffer
        @test IOM._block_offer_constraint(IOM.IncrementalOffer()) ==
              IOM.PiecewiseLinearBlockIncrementalOfferConstraint
        @test IOM._block_offer_constraint(IOM.DecrementalOffer()) ==
              IOM.PiecewiseLinearBlockDecrementalOfferConstraint
        @test IOM._objective_sign(IOM.IncrementalOffer()) ≈ IOM.OBJECTIVE_FUNCTION_POSITIVE
        @test IOM._objective_sign(IOM.DecrementalOffer()) ≈ IOM.OBJECTIVE_FUNCTION_NEGATIVE
        @test IOM._slope_param(IOM.IncrementalOffer()) ==
              IOM.IncrementalPiecewiseLinearSlopeParameter
        @test IOM._slope_param(IOM.DecrementalOffer()) ==
              IOM.DecrementalPiecewiseLinearSlopeParameter
        @test IOM._breakpoint_param(IOM.IncrementalOffer()) ==
              IOM.IncrementalPiecewiseLinearBreakpointParameter
        @test IOM._breakpoint_param(IOM.DecrementalOffer()) ==
              IOM.DecrementalPiecewiseLinearBreakpointParameter
    end
end
