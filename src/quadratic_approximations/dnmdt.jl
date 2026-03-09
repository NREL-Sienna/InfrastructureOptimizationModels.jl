# D-NMDT and T-D-NMDT (Doubly Discretized Normalized Multiparametric Disaggregation Technique)
# MIP relaxations for bilinear products z = x*y and univariate quadratics z = x^2.
# A single `tighten` flag controls whether epigraph cuts are added (T-D-NMDT) or not (D-NMDT).
# Reference: Beach, Burlacu, Barmann, Hager, Hildebrand (2024), Definitions 8-10.

# ── Variable types ────────────────────────────────────────────────────────────
"Binary expansion variables beta[j] in D-NMDT base-2 discretization."
struct DNMDTBinaryVariable <: VariableType end
"Continuous residual Delta_w in D-NMDT base-2 discretization."
struct DNMDTResidualVariable <: VariableType end
"Scaled [0,1] variable w_hat = (w - w_min) / (w_max - w_min)."
struct DNMDTScaledVariable <: VariableType end
"Product auxiliary variables u[j], v[j] from binary x continuous McCormick."
struct DNMDTProductAuxVariable <: VariableType end
"Residual product variable Delta_z from McCormick on Delta_x * Delta_y."
struct DNMDTResidualProductVariable <: VariableType end

# ── Constraint types ──────────────────────────────────────────────────────────
"Binary expansion constraint: w_hat = sum(2^-j * beta[j]) + Delta_w."
struct DNMDTBinaryExpansionConstraint <: ConstraintType end
"Scaling constraint: w_hat = (w - w_min) / (w_max - w_min)."
struct DNMDTScalingConstraint <: ConstraintType end
"McCormick constraints on beta[j] x continuous products."
struct DNMDTBinaryMcCormickConstraint <: ConstraintType end
"Back-transformation constraint: z = lx*ly*z_hat + offsets."
struct DNMDTBackTransformConstraint <: ConstraintType end
"Upper-bound McCormick on residual product (tightened univariate case)."
struct DNMDTResidualUpperBoundConstraint <: ConstraintType end
"McCormick bounds on x^2 (global convex/concave envelope)."
struct DNMDTSquareBoundConstraint <: ConstraintType end

# ── Expression types ──────────────────────────────────────────────────────────
"Blended continuous terms (w_u, w_v, w_sum) used in binary McCormick."
struct DNMDTBlendedTermExpression <: ExpressionType end
"Scaled product z_hat expression."
struct DNMDTScaledProductExpression <: ExpressionType end
"Final bivariate result expression z ~ x*y."
struct DNMDTBilinearExpression <: ExpressionType end
"Final univariate result expression z ~ x^2."
struct DNMDTQuadraticExpression <: ExpressionType end

# ── Helper: populate binary expansion ─────────────────────────────────────────

"""
    _populate_binary_expansion!(jump_model, C, names, time_steps, w_var_container, w_min, lw, depth, eps_L, wh_con, beta_con, dw_con, scaling_con, expansion_con)

Populate pre-allocated containers with base-2 binary expansion variables and constraints.

For each (name, t), creates:
- Scaled variable w_hat in [0,1]
- L binary variables beta[j]
- Continuous residual Delta_w in [0, 2^{-L}]
- Scaling constraint: w_hat = (w - w_min) / lw
- Expansion constraint: w_hat = sum(2^{-j} * beta[j]) + Delta_w
"""
function _populate_binary_expansion!(
    jump_model::JuMP.Model,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    w_var_container,
    w_min::Float64,
    lw::Float64,
    depth::Int,
    eps_L::Float64,
    wh_con,
    beta_con,
    dw_con,
    scaling_con,
    expansion_con,
) where {C <: IS.InfrastructureSystemsComponent}
    for name in names, t in time_steps
        wh_con[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTScaled_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = 1.0,
        )
        dw_con[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTResidual_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = eps_L,
        )
        for j in 1:depth
            beta_con[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTBin_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
        end
        scaling_con[name, t] = JuMP.@constraint(
            jump_model,
            wh_con[name, t] == (w_var_container[name, t] - w_min) / lw,
        )
        expansion_con[name, t] = JuMP.@constraint(
            jump_model,
            wh_con[name, t] ==
            sum(2.0^(-j) * beta_con[name, j, t] for j in 1:depth) + dw_con[name, t],
        )
    end
    return
end

# ── Univariate D-NMDT / T-D-NMDT for z = x^2 ────────────────────────────────

"""
    _add_dnmdt_univariate_approx!(container, C, names, time_steps, x_var_container, x_min, x_max, depth, meta; tighten, epigraph_depth)

Approximate z = x^2 using univariate D-NMDT (Definition 9) or T-D-NMDT (Definition 10).

When `tighten=false`: D-NMDT with full residual McCormick.
When `tighten=true`: T-D-NMDT replaces residual lower bounds with epigraph Q^{L1} cuts.

Stores result in a `DNMDTQuadraticExpression` expression container.
"""
function _add_dnmdt_univariate_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = true,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth)),
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op depth >= 1
    jump_model = get_jump_model(container)
    lx = x_max - x_min
    eps_L = 2.0^(-depth)
    ws_hi = eps_L + 1.0

    # ── Allocate containers ──────────────────────────────────────────────

    # Binary expansion
    xh_con = add_variable_container!(
        container, DNMDTScaledVariable(), C, names, time_steps; meta = meta,
    )
    beta_con = add_variable_container!(
        container, DNMDTBinaryVariable(), C, names, 1:depth, time_steps; meta = meta,
    )
    dx_con = add_variable_container!(
        container, DNMDTResidualVariable(), C, names, time_steps; meta = meta,
    )
    scaling = add_constraints_container!(
        container, DNMDTScalingConstraint(), C, names, time_steps; meta = meta,
    )
    expansion = add_constraints_container!(
        container, DNMDTBinaryExpansionConstraint(), C, names, time_steps; meta = meta,
    )

    # Sum term w_sum = Delta_x + x_hat as expression
    ws_expr = add_expression_container!(
        container, DNMDTBlendedTermExpression(), C, names, time_steps;
        meta = meta * "_ws",
    )

    # Product aux u[j]
    u_con = add_variable_container!(
        container, DNMDTProductAuxVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_u",
    )
    bmc = add_constraints_container!(
        container, DNMDTBinaryMcCormickConstraint(), C,
        names, 1:3, 1:depth, time_steps; sparse = true, meta = meta,
    )

    # Residual Delta_z
    dz_con = add_variable_container!(
        container, DNMDTResidualProductVariable(), C, names, time_steps; meta = meta,
    )

    # Scaled product z_hat as expression
    zh_expr = add_expression_container!(
        container, DNMDTScaledProductExpression(), C, names, time_steps; meta = meta,
    )

    # Final z variable + back-transform
    z_con = add_variable_container!(
        container, BilinearProductVariable(), C, names, time_steps; meta = meta,
    )
    bt = add_constraints_container!(
        container, DNMDTBackTransformConstraint(), C, names, time_steps; meta = meta,
    )

    # McCormick on x^2 (global bounds)
    sq = add_constraints_container!(
        container, DNMDTSquareBoundConstraint(), C,
        names, 1:3, time_steps; sparse = true, meta = meta,
    )

    # Result expression
    result_expr = add_expression_container!(
        container, DNMDTQuadraticExpression(), C, names, time_steps; meta = meta,
    )

    # Tightening: residual upper bound
    if tighten
        ub = add_constraints_container!(
            container, DNMDTResidualUpperBoundConstraint(), C,
            names, time_steps; meta = meta,
        )
    end

    # ── Populate: binary expansion ───────────────────────────────────────

    _populate_binary_expansion!(
        jump_model, C, names, time_steps,
        x_var_container, x_min, lx, depth, eps_L,
        xh_con, beta_con, dx_con, scaling, expansion,
    )

    # ── Populate: sum term, product aux, assembly, back-transform ────────

    for name in names, t in time_steps
        x = x_var_container[name, t]

        # Sum term as expression: w_sum = Delta_x + x_hat
        w_sum = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(w_sum, 1.0, dx_con[name, t])
        JuMP.add_to_expression!(w_sum, 1.0, xh_con[name, t])
        ws_expr[name, t] = w_sum

        # Binary McCormick for u[j]
        for j in 1:depth
            u_con[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTu_$(C)_{$(name), $(j), $(t)}",
                lower_bound = 0.0,
            )
            u_j = u_con[name, j, t]
            beta_j = beta_con[name, j, t]
            bmc[(name, 1, j, t)] =
                JuMP.@constraint(jump_model, u_j <= ws_hi * beta_j)
            bmc[(name, 2, j, t)] =
                JuMP.@constraint(jump_model, u_j >= w_sum - ws_hi * (1 - beta_j))
            bmc[(name, 3, j, t)] =
                JuMP.@constraint(jump_model, u_j <= w_sum)
        end

        # Residual
        dz_con[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTdz_$(C)_{$(name), $(t)}",
        )
        if tighten
            ub[name, t] = JuMP.@constraint(
                jump_model,
                dz_con[name, t] <= eps_L * dx_con[name, t],
            )
        end

        # Scaled product z_hat as expression
        zh = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(zh, 2.0^(-j), u_con[name, j, t])
        end
        JuMP.add_to_expression!(zh, 1.0, dz_con[name, t])
        zh_expr[name, t] = zh

        # Back-transform: z = lx^2 * z_hat + 2 * x_min * lx * x_hat + x_min^2
        z_var = JuMP.@variable(
            jump_model,
            base_name = "DNMDTz_$(C)_{$(name), $(t)}",
        )
        z_con[name, t] = z_var
        bt[name, t] = JuMP.@constraint(
            jump_model,
            z_var == lx^2 * zh + 2 * x_min * lx * xh_con[name, t] + x_min^2,
        )
        result_expr[name, t] = JuMP.AffExpr(0.0, z_var => 1.0)

        # McCormick on x^2
        sq[(name, 1, t)] =
            JuMP.@constraint(jump_model, z_var >= 2 * x_min * x - x_min^2)
        sq[(name, 2, t)] =
            JuMP.@constraint(jump_model, z_var >= 2 * x_max * x - x_max^2)
        sq[(name, 3, t)] =
            JuMP.@constraint(jump_model, z_var <= (x_min + x_max) * x - x_min * x_max)
    end

    # ── Residual McCormick (D-NMDT only) ──
    if !tighten
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            dx_con, dx_con, dz_con,
            0.0, eps_L, 0.0, eps_L, meta * "_residual",
        )
    end

    # ── Epigraph tightening (T-D-NMDT only) ──
    if tighten
        _add_epigraph_quadratic_approx!(
            container, C, names, time_steps,
            x_var_container, x_min, x_max, epigraph_depth, meta * "_epi",
        )
        epi_container = get_expression(
            container, EpigraphExpression(), C, meta * "_epi",
        )
        epi_lb = add_constraints_container!(
            container, DNMDTSquareBoundConstraint(), C,
            names, time_steps; meta = meta * "_epi_lb",
        )
        for name in names, t in time_steps
            epi_lb[name, t] = JuMP.@constraint(
                jump_model,
                z_con[name, t] >= epi_container[name, t],
            )
        end
    end

    return
end

# ── Bivariate D-NMDT for z = x*y ─────────────────────────────────────────────

"""
    _add_dnmdt_bilinear_approx!(container, C, names, time_steps, x_var_container, y_var_container, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate z = x*y using bivariate D-NMDT (Definition 8).

Discretizes both x and y with base-2 binary expansion, applies McCormick envelopes
to all binary x continuous products, and handles the residual product via McCormick.
Uses lambda = DNMDT_LAMBDA (0.5) per Remark 1.

Stores result in a `DNMDTBilinearExpression` expression container.
"""
function _add_dnmdt_bilinear_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    y_var_container,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    add_mccormick::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op y_max > y_min
    IS.@assert_op depth >= 1
    jump_model = get_jump_model(container)
    lx = x_max - x_min
    ly = y_max - y_min
    eps_L = 2.0^(-depth)
    meta_x = meta * "_x"
    meta_y = meta * "_y"

    # Blended-term bounds
    lambda = DNMDT_LAMBDA
    wu_lo = 0.0
    wu_hi = lambda * eps_L + (1 - lambda)
    wv_lo = 0.0
    wv_hi = (1 - lambda) * eps_L + lambda

    # ── Allocate containers ──────────────────────────────────────────────

    # Binary expansion of x
    xh_con = add_variable_container!(
        container, DNMDTScaledVariable(), C, names, time_steps; meta = meta_x,
    )
    beta_x_con = add_variable_container!(
        container, DNMDTBinaryVariable(), C, names, 1:depth, time_steps; meta = meta_x,
    )
    dx_con = add_variable_container!(
        container, DNMDTResidualVariable(), C, names, time_steps; meta = meta_x,
    )
    scaling_x = add_constraints_container!(
        container, DNMDTScalingConstraint(), C, names, time_steps; meta = meta_x,
    )
    expansion_x = add_constraints_container!(
        container, DNMDTBinaryExpansionConstraint(), C, names, time_steps; meta = meta_x,
    )

    # Binary expansion of y
    yh_con = add_variable_container!(
        container, DNMDTScaledVariable(), C, names, time_steps; meta = meta_y,
    )
    beta_y_con = add_variable_container!(
        container, DNMDTBinaryVariable(), C, names, 1:depth, time_steps; meta = meta_y,
    )
    dy_con = add_variable_container!(
        container, DNMDTResidualVariable(), C, names, time_steps; meta = meta_y,
    )
    scaling_y = add_constraints_container!(
        container, DNMDTScalingConstraint(), C, names, time_steps; meta = meta_y,
    )
    expansion_y = add_constraints_container!(
        container, DNMDTBinaryExpansionConstraint(), C, names, time_steps; meta = meta_y,
    )

    # Blended terms as expressions
    wu_expr = add_expression_container!(
        container, DNMDTBlendedTermExpression(), C, names, time_steps;
        meta = meta * "_wu",
    )
    wv_expr = add_expression_container!(
        container, DNMDTBlendedTermExpression(), C, names, time_steps;
        meta = meta * "_wv",
    )

    # Product aux variables u[j], v[j]
    u_con = add_variable_container!(
        container, DNMDTProductAuxVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_u",
    )
    v_con = add_variable_container!(
        container, DNMDTProductAuxVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_v",
    )

    # Binary McCormick constraints
    bmc = add_constraints_container!(
        container, DNMDTBinaryMcCormickConstraint(), C,
        names, 1:3, 1:depth, ["u", "v"], time_steps; sparse = true, meta = meta,
    )

    # Residual product Delta_z
    dz_con = add_variable_container!(
        container, DNMDTResidualProductVariable(), C, names, time_steps; meta = meta,
    )

    # Scaled product z_hat as expression
    zh_expr = add_expression_container!(
        container, DNMDTScaledProductExpression(), C, names, time_steps; meta = meta,
    )

    # Final z variable + back-transform constraint
    z_con = add_variable_container!(
        container, BilinearProductVariable(), C, names, time_steps; meta = meta,
    )
    bt = add_constraints_container!(
        container, DNMDTBackTransformConstraint(), C, names, time_steps; meta = meta,
    )

    # Result expression
    result_expr = add_expression_container!(
        container, DNMDTBilinearExpression(), C, names, time_steps; meta = meta,
    )

    # ── Populate: binary expansions ──────────────────────────────────────

    _populate_binary_expansion!(
        jump_model, C, names, time_steps,
        x_var_container, x_min, lx, depth, eps_L,
        xh_con, beta_x_con, dx_con, scaling_x, expansion_x,
    )

    _populate_binary_expansion!(
        jump_model, C, names, time_steps,
        y_var_container, y_min, ly, depth, eps_L,
        yh_con, beta_y_con, dy_con, scaling_y, expansion_y,
    )

    # ── Populate: blended terms, product aux, assembly, back-transform ───

    for name in names, t in time_steps
        # Blended terms: w_u = lambda*Delta_y + (1-lambda)*y_hat
        w_u = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(w_u, lambda, dy_con[name, t])
        JuMP.add_to_expression!(w_u, 1 - lambda, yh_con[name, t])
        wu_expr[name, t] = w_u

        # w_v = (1-lambda)*Delta_x + lambda*x_hat
        w_v = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(w_v, 1 - lambda, dx_con[name, t])
        JuMP.add_to_expression!(w_v, lambda, xh_con[name, t])
        wv_expr[name, t] = w_v

        # Binary McCormick for u[j] and v[j]
        for j in 1:depth
            u_con[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTu_$(C)_{$(name), $(j), $(t)}",
                lower_bound = 0.0,
            )
            v_con[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTv_$(C)_{$(name), $(j), $(t)}",
                lower_bound = 0.0,
            )

            u_j = u_con[name, j, t]
            beta_xj = beta_x_con[name, j, t]
            bmc[(name, 1, j, "u", t)] =
                JuMP.@constraint(jump_model, u_j <= wu_hi * beta_xj)
            bmc[(name, 2, j, "u", t)] =
                JuMP.@constraint(jump_model, u_j >= w_u - wu_hi * (1 - beta_xj))
            bmc[(name, 3, j, "u", t)] =
                JuMP.@constraint(jump_model, u_j <= w_u)

            v_j = v_con[name, j, t]
            beta_yj = beta_y_con[name, j, t]
            bmc[(name, 1, j, "v", t)] =
                JuMP.@constraint(jump_model, v_j <= wv_hi * beta_yj)
            bmc[(name, 2, j, "v", t)] =
                JuMP.@constraint(jump_model, v_j >= w_v - wv_hi * (1 - beta_yj))
            bmc[(name, 3, j, "v", t)] =
                JuMP.@constraint(jump_model, v_j <= w_v)
        end

        # Residual product variable
        dz_con[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTdz_$(C)_{$(name), $(t)}",
        )

        # Scaled product z_hat as expression
        zh = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(zh, 2.0^(-j), u_con[name, j, t])
            JuMP.add_to_expression!(zh, 2.0^(-j), v_con[name, j, t])
        end
        JuMP.add_to_expression!(zh, 1.0, dz_con[name, t])
        zh_expr[name, t] = zh

        # Back-transform: z = lx*ly*z_hat + x_min*ly*y_hat + y_min*lx*x_hat + x_min*y_min
        z_var = JuMP.@variable(
            jump_model,
            base_name = "DNMDTz_$(C)_{$(name), $(t)}",
        )
        z_con[name, t] = z_var
        bt[name, t] = JuMP.@constraint(
            jump_model,
            z_var ==
            lx * ly * zh +
            x_min * ly * yh_con[name, t] +
            y_min * lx * xh_con[name, t] +
            x_min * y_min,
        )
        result_expr[name, t] = JuMP.AffExpr(0.0, z_var => 1.0)
    end

    # ── Residual McCormick (Delta_z ~ Delta_x * Delta_y) ──
    _add_mccormick_envelope!(
        container, C, names, time_steps,
        dx_con, dy_con, dz_con,
        0.0, eps_L, 0.0, eps_L, meta * "_residual",
    )

    # ── Global McCormick on z = x*y ──
    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var_container, y_var_container, z_con,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    return
end
