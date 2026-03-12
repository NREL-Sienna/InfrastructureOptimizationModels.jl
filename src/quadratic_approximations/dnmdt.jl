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
struct DNMDTBinaryExpansionExpression <: ExpressionType end
"Scaling constraint: w_hat = (w - w_min) / (w_max - w_min)."
struct DNMDTScalingConstraint <: ConstraintType end
"McCormick constraints on beta[j] x continuous products."
struct DNMDTBinaryMcCormickConstraint <: ConstraintType end
"Upper-bound McCormick on residual product (tightened univariate case)."
struct DNMDTResidualUpperBoundConstraint <: ConstraintType end
"McCormick bounds on x^2 (global convex/concave envelope)."
struct DNMDTSquareBoundConstraint <: ConstraintType end

# ── Helper: populate binary expansion ─────────────────────────────────────────

"""
    _populate_binary_expansion!(jump_model, C, names, time_steps, x_var, x_min, lw, depth, eps_L, xh_var, beta_var, dx_var, scaling_con, expansion_con)

Populate pre-allocated containers with base-2 binary expansion variables and constraints.

For each (name, t), creates:
- Scaled variable x_hat in [0,1]
- L binary variables beta[j]
- Continuous residual Delta_x in [0, 2^{-L}]
- Scaling constraint: x_hat = (x - x_min) / lx
- Expansion constraint: x_hat = sum(2^{-j} * beta[j]) + Delta_x
"""
function _populate_binary_expansion!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    lx::Float64,
    depth::Int,
    eps_L::Float64,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    xh_var = @_add_container!(variable, DNMDTScaledVariable)
    beta_var = @_add_container!(variable, DNMDTBinaryVariable, 1:depth)
    delta_var = @_add_container!(variable, DNMDTResidualVariable)
    scaling_cons = @_add_container!(constraints, DNMDTScalingConstraint)
    expansion_cons = @_add_container!(constraints, DNMDTBinaryExpansionConstraint)
    expansion_expr = @_add_container!(expression, DNMDTBinaryExpansionExpression)

    for name in names, t in time_steps
        xh =
            xh_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTScaled_$(C)_{$(name), $(t)}",
                lower_bound = 0.0,
                upper_bound = 1.0,
            )

        delta =
            delta_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTResidual_$(C)_{$(name), $(t)}",
                lower_bound = 0.0,
                upper_bound = eps_L,
            )

        for j in 1:depth
            beta_var[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTBin_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
        end
        scaling_cons[name, t] = JuMP.@constraint(
            jump_model,
            xh == (x_var[name, t] - x_min) / lx,
        )
        ex = expansion_expr[name, t] = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(ex, 2.0^(-j), beta_var[name, j, t])
        end
        JuMP.add_to_expression!(ex, delta)

        expansion_cons[name, t] = JuMP.@constraint(jump_model, xh == ex)
    end
    return xh_var, beta_var, delta_var
end

# ── Univariate D-NMDT / T-D-NMDT for z = x^2 ────────────────────────────────

struct WSumExpression <: ExpressionType end

"""
    _add_dnmdt_univariate_approx!(container, C, names, time_steps, x_var, x_min, x_max, depth, meta; tighten, epigraph_depth)

Approximate z = x^2 using univariate D-NMDT (Definition 9) or T-D-NMDT (Definition 10).

When `tighten=false`: D-NMDT with full residual McCormick.
When `tighten=true`: T-D-NMDT replaces residual lower bounds with epigraph Q^{L1} cuts.

Stores result in a `QuadraticExpression` expression container.
"""
function _add_dnmdt_univariate_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
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
    meta_u = meta * "_u"

    # Precompute power-of-two coefficients (invariant across names and time steps)
    pow2_neg = [2.0^(-j) for j in 1:depth]

    # ── Allocate containers ──────────────────────────────────────────────

    u_var = @_add_container!(variable, DNMDTProductAuxVariable, 1:depth, meta_u)
    bmc_cons =
        @_add_container!(constraints, DNMDTBinaryMcCormickConstraint, 1:depth, 1:4, sparse)
    dz_var = @_add_container!(variable, DNMDTResidualProductVariable)
    result_expr = @_add_container!(expression, QuadraticExpression)
    if tighten
        ub_cons = @_add_container!(constraints, DNMDTResidualUpperBoundConstraint)
    end

    # ── Populate: binary expansion ───────────────────────────────────────

    xh_var, beta_var, dx_var = _populate_binary_expansion!(
        container, C, names, time_steps,
        x_var, x_min, lx, depth, eps_L, meta,
    )

    # ── Populate: sum term, product aux, assembly, back-transform ────────

    for name in names, t in time_steps
        # Sum term: w_sum = Delta_x + x_hat (used locally in McCormick below)
        w_sum = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(w_sum, dx_var[name, t])
        JuMP.add_to_expression!(w_sum, xh_var[name, t])

        # Binary McCormick for u[j]
        for j in 1:depth
            u_j =
                u_var[name, j, t] = JuMP.@variable(
                    jump_model,
                    base_name = "DNMDTu_$(C)_{$(name), $(j), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = ws_hi,
                )
            _add_mccormick_envelope!(
                jump_model, bmc_cons, (name, j, t),
                w_sum, beta_var[name, j, t], u_j,
                0.0, ws_hi, 0.0, 1.0,
            )
        end

        # Residual (bounded by product of residual ranges [0, eps_L]²)
        dz_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTdz_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = eps_L * eps_L,
        )
        if tighten
            ub_cons[name, t] = JuMP.@constraint(
                jump_model,
                dz_var[name, t] <= eps_L * dx_var[name, t],
            )
        end

        # Scaled product z_hat (used locally in back-transform below)
        zh = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(zh, pow2_neg[j], u_var[name, j, t])
        end
        JuMP.add_to_expression!(zh, dz_var[name, t])

        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lx^2, zh)
        JuMP.add_to_expression!(result, 2 * x_min * lx, xh_var[name, t])
        JuMP.add_to_expression!(result, x_min^2)
    end

    # Global McCormick
    _add_mccormick_envelope!(
        container, C, names, time_steps,
        x_var, result_expr,
        x_min, x_max, meta
    )

    # ── Residual McCormick (D-NMDT only) ──
    if !tighten
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            dx_var, dz_var,
            0.0, eps_L, meta * "_residual",
        )

        # ── Epigraph tightening (T-D-NMDT only) ──
    else
        epi_expr = _add_epigraph_quadratic_approx!(
            container, C, names, time_steps,
            x_var, x_min, x_max, epigraph_depth, meta * "_epi",
        )
        meta_lb = meta * "_epi_lb"
        epi_cons = @_add_container!(constraints, DNMDTSquareBoundConstraint, meta_lb)
        for name in names, t in time_steps
            epi_cons[name, t] = JuMP.@constraint(
                jump_model,
                result_expr[name, t] >= epi_expr[name, t],
            )
        end
    end

    return result_expr
end

# ── Bivariate D-NMDT for z = x*y ─────────────────────────────────────────────

"""
    _add_dnmdt_bilinear_approx!(container, C, names, time_steps, x_var, x_var, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate z = x*y using bivariate D-NMDT (Definition 8).

Discretizes both x and y with base-2 binary expansion, applies McCormick envelopes
to all binary x continuous products, and handles the residual product via McCormick.
Uses lambda = DNMDT_LAMBDA (0.5) per Remark 1.

Stores result in a `BilinearProductExpression` expression container.
"""
function _add_dnmdt_bilinear_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
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
    meta_u = meta * "u"
    meta_v = meta * "v"
    meta_x = meta * "_x"
    meta_y = meta * "_y"

    # Precompute power-of-two coefficients
    pow2_neg = [2.0^(-j) for j in 1:depth]

    # Blended-term bounds
    lambda = DNMDT_LAMBDA
    wu_hi = lambda * eps_L + (1 - lambda)
    wv_hi = (1 - lambda) * eps_L + lambda

    # ── Allocate containers ──────────────────────────────────────────────

    # Product aux variables u[j], v[j]
    u_var = @_add_container!(variable, DNMDTProductAuxVariable, 1:depth, meta_u)
    v_var = @_add_container!(variable, DNMDTProductAuxVariable, 1:depth, meta_v)
    bmc_cons = @_add_container!(
        constraints,
        DNMDTBinaryMcCormickConstraint,
        1:3,
        1:depth,
        ["u", "v"],
        sparse
    )
    dz_var = @_add_container!(variable, DNMDTResidualProductVariable)
    result_expr = @_add_container!(expression, BilinearProductExpression)

    # ── Populate: binary expansions ──────────────────────────────────────

    xh_var, beta_x_var, dx_var = _populate_binary_expansion!(
        container, C, names, time_steps,
        x_var, x_min, lx, depth, eps_L, meta_x,
    )
    yh_var, beta_y_var, dy_var = _populate_binary_expansion!(
        container, C, names, time_steps,
        y_var, y_min, ly, depth, eps_L, meta_y,
    )

    # ── Populate: blended terms, product aux, assembly, back-transform ───

    for name in names, t in time_steps
        # Blended terms (used locally in McCormick below)
        w_u = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(w_u, lambda, dy_var[name, t])
        JuMP.add_to_expression!(w_u, 1 - lambda, yh_var[name, t])

        w_v = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(w_v, 1 - lambda, dx_var[name, t])
        JuMP.add_to_expression!(w_v, lambda, xh_var[name, t])

        # Binary McCormick for u[j] and v[j]
        for j in 1:depth
            u_j =
                u_var[name, j, t] = JuMP.@variable(
                    jump_model,
                    base_name = "DNMDTu_$(C)_{$(name), $(j), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = wu_hi,
                )
            v_j =
                v_var[name, j, t] = JuMP.@variable(
                    jump_model,
                    base_name = "DNMDTv_$(C)_{$(name), $(j), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = wv_hi,
                )

            beta_xj = beta_x_var[name, j, t]
            bmc_cons[(name, 1, j, "u", t)] =
                JuMP.@constraint(jump_model, u_j <= wu_hi * beta_xj)
            bmc_cons[(name, 2, j, "u", t)] =
                JuMP.@constraint(jump_model, u_j >= w_u - wu_hi * (1 - beta_xj))
            bmc_cons[(name, 3, j, "u", t)] =
                JuMP.@constraint(jump_model, u_j <= w_u)

            beta_yj = beta_y_var[name, j, t]
            bmc_cons[(name, 1, j, "v", t)] =
                JuMP.@constraint(jump_model, v_j <= wv_hi * beta_yj)
            bmc_cons[(name, 2, j, "v", t)] =
                JuMP.@constraint(jump_model, v_j >= w_v - wv_hi * (1 - beta_yj))
            bmc_cons[(name, 3, j, "v", t)] =
                JuMP.@constraint(jump_model, v_j <= w_v)
        end

        # Residual product variable (bounded by [0, eps_L²])
        dz_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTdz_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = eps_L * eps_L,
        )

        # Scaled product z_hat (used locally in back-transform below)
        zh = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(zh, pow2_neg[j], u_var[name, j, t])
            JuMP.add_to_expression!(zh, pow2_neg[j], v_var[name, j, t])
        end
        JuMP.add_to_expression!(zh, dz_var[name, t])

        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lx * ly, zh)
        JuMP.add_to_expression!(result, x_min * ly, yh_var[name, t])
        JuMP.add_to_expression!(result, y_min * lx, xh_var[name, t])
        JuMP.add_to_expression!(result, x_min * y_min)
    end

    # ── Residual McCormick (Delta_z ~ Delta_x * Delta_y) ──
    _add_mccormick_envelope!(
        container, C, names, time_steps,
        dx_var, dy_var, dz_var,
        0.0, eps_L, 0.0, eps_L, meta * "_residual",
    )

    # ── Global McCormick on z = x*y ──
    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, result_expr,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    return
end
