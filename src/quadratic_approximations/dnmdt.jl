# D-NMDT and T-D-NMDT (Doubly Discretized Normalized Multiparametric Disaggregation Technique)
# MIP relaxations for bilinear products z = x*y and univariate quadratics z = x^2.
# A single `tighten` flag controls whether epigraph cuts are added (T-D-NMDT) or not (D-NMDT).
# Reference: Beach, Burlacu, Barmann, Hager, Hildebrand (2024), Definitions 8-10.

# ── Variable types ────────────────────────────────────────────────────────────
"Binary expansion variables beta[j] in D-NMDT base-2 discretization."
struct DNMDTBinaryVariable <: VariableType end
"Continuous residual Delta_w in D-NMDT base-2 discretization."
struct DNMDTResidualVariable <: VariableType end
"Product auxiliary variables u[j], v[j] from binary x continuous McCormick."
struct DNMDTProductAuxVariable <: VariableType end
"Residual product variable Delta_z from McCormick on Delta_x * Delta_y."
struct DNMDTResidualProductVariable <: VariableType end

# ── Constraint types ──────────────────────────────────────────────────────────
"Binary expansion constraint: w_hat = sum(2^-j * beta[j]) + Delta_w."
struct DNMDTBinaryExpansionConstraint <: ConstraintType end
"McCormick constraints on beta[j] x continuous products."
struct DNMDTBinaryMcCormickConstraint <: ConstraintType end
"Upper-bound McCormick on residual product (tightened univariate case)."
struct DNMDTResidualUpperBoundConstraint <: ConstraintType end
"McCormick bounds on x^2 (global convex/concave envelope)."
struct DNMDTSquareBoundConstraint <: ConstraintType end

# ── Expression types ──────────────────────────────────────────────────────────
"Scaled expression: w_hat = (w - w_min) / (w_max - w_min)."
struct DNMDTScaledVariableExpression <: ExpressionType end
"Binary expansion constraint: w_hat = sum(2^-j * beta[j]) + Delta_w."
struct DNMDTBinaryExpansionExpression <: ExpressionType end
"Intermediate sum before McCormick: w_sum = x_hat + delta_x"
struct DNMDTIntermediateSumExpression <: ExpressionType end
"Scaled product expression z_hat"
struct DNMDTScaledProductExpression <: ExpressionType end

# ── Helper: populate binary expansion ─────────────────────────────────────────

# Will refactor out later
function _scale!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    lx::Float64,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        DNMDTScaledVariableExpression(),
        C,
        names,
        time_steps;
        meta
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(-x_min / lx)
        JuMP.add_to_expression!(result, 1.0 / lx, x_var[name, t])
    end
    return result_expr
end

"""
    _populate_binary_expansion!(jump_model, C, names, time_steps, x_var, x_min, lx, depth, meta)

Creates and populates containers with base-2 binary expansion variables and constraints.

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
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    xh_expr = _scale!(
        container, C, names, time_steps,
        x_var, x_min, lx,
        meta
    )
    beta_var = add_variable_container!(
        container,
        DNMDTBinaryVariable(),
        C,
        names,
        1:depth,
        time_steps;
        meta,
    )
    delta_var = add_variable_container!(
        container,
        DNMDTResidualVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    expansion_expr = add_expression_container!(
        container,
        DNMDTBinaryExpansionExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    expansion_cons = add_constraints_container!(
        container,
        DNMDTBinaryExpansionConstraint(),
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        delta =
            delta_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTResidual_$(C)_{$(name), $(t)}",
                lower_bound = 0.0,
                upper_bound = 2.0^(-depth),
            )

        for j in 1:depth
            beta_var[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "DNMDTBin_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
        end

        expansion = expansion_expr[name, t] = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(expansion, 2.0^(-j), beta_var[name, j, t])
        end
        JuMP.add_to_expression!(expansion, delta)
        expansion_cons[name, t] = JuMP.@constraint(jump_model, xh_expr[name, t] == expansion)
    end
    return xh_expr, beta_var, delta_var
end

# ── Univariate D-NMDT / T-D-NMDT for z = x^2 ────────────────────────────────

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
    double::Bool = true,
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op depth >= 1
    jump_model = get_jump_model(container)
    lx = x_max - x_min
    eps_L = 2.0^(-depth)

    # ── Allocate containers ──────────────────────────────────────────────

    wsum_expr = add_expression_container!(
        container,
        DNMDTIntermediateSumExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    u_var = add_variable_container!(
        container,
        DNMDTProductAuxVariable(),
        C,
        names,
        1:depth,
        time_steps;
        meta = meta * "_u",
    )
    bmc_cons = add_constraints_container!(
        container,
        DNMDTBinaryMcCormickConstraint(),
        C,
        names,
        1:depth,
        1:4,
        time_steps;
        sparse = true,
        meta,
    )
    dz_var = add_variable_container!(
        container,
        DNMDTResidualProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    zh_expr = add_expression_container!(
        container,
        DNMDTScaledProductExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    result_expr = add_expression_container!(
        container,
        QuadraticExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    # ── Populate: binary expansion ───────────────────────────────────────

    xh_expr, beta_var, dx_var = _populate_binary_expansion!(
        container, C, names, time_steps,
        x_var, x_min, lx, depth, meta,
    )

    # ── Populate: sum term, product aux, assembly, back-transform ────────

    for name in names, t in time_steps
        # Sum term: w_sum = Delta_x + x_hat (used locally in McCormick below)
        if double
            w_sum = wsum_expr[name, t] = JuMP.AffExpr(0.0)
            JuMP.add_to_expression!(w_sum, dx_var[name, t])
            JuMP.add_to_expression!(w_sum, xh_expr[name, t])
            ws_hi = eps_L + 1.0
        else
            w_sum = xh_expr[name, t]
            ws_hi = 1.0
        end

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
                0.0, ws_hi, 0.0, 1.0;
                lower_bounds = !tighten,
            )
        end

        # Residual (bounded by product of residual ranges [0, eps_L]²)
        dz_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTdz_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = eps_L * eps_L,
        )

        # Scaled product z_hat (used locally in back-transform below)
        zh = zh_expr[name, t] = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(zh, 2.0^(-j), u_var[name, j, t])
        end
        JuMP.add_to_expression!(zh, dz_var[name, t])

        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lx^2, zh)
        JuMP.add_to_expression!(result, 2 * x_min * lx, xh_expr[name, t])
        JuMP.add_to_expression!(result, x_min^2)
    end

    # ── Residual McCormick ──
    if double
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            dx_var, dz_var,
            0.0, eps_L, meta * "_residual";
            lower_bounds = !tighten,
        )
    else
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            dx_var, xh_expr, dz_var,
            0.0, eps_L, 0.0, 1.0,
            meta * "_residual";
            lower_bounds = !tighten,
        )
    end

    # ── Epigraph tightening (T-D-NMDT only) ──
    if tighten
        epi_expr = _add_epigraph_quadratic_approx!(
            container, C, names, time_steps,
            x_var, x_min, x_max, epigraph_depth, meta * "_epi",
        )
        epi_cons = add_constraints_container!(
            container,
            DNMDTSquareBoundConstraint(),
            C,
            names,
            time_steps;
            meta = meta * "_epi_lb",
        )
        for name in names, t in time_steps
            epi_cons[name, t] = JuMP.@constraint(
                jump_model,
                result_expr[name, t] >= epi_expr[name, t],
            )
        end
    end

    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, result_expr,
            x_min, x_max, y_min, y_max, meta
        )
    end

    return result_expr
end

# ── Bivariate D-NMDT for z = x*y ─────────────────────────────────────────────

"""
    _add_dnmdt_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

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
    double::Bool = true,
    add_mccormick::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op y_max > y_min
    IS.@assert_op depth >= 1
    jump_model = get_jump_model(container)
    lx = x_max - x_min
    ly = y_max - y_min
    eps_L = 2.0^(-depth)

    # Precompute power-of-two coefficients
    pow2_neg = [2.0^(-j) for j in 1:depth]

    # Blended-term bounds
    lambda = DNMDT_LAMBDA
    wu_hi = lambda * eps_L + (1 - lambda)
    wv_hi = (1 - lambda) * eps_L + lambda

    # ── Allocate containers ──────────────────────────────────────────────

    u_var = add_variable_container!(
        container,
        DNMDTProductAuxVariable(),
        C,
        names,
        1:depth,
        time_steps;
        meta = meta * "_u",
    )
    if double
        v_var = add_variable_container!(
            container,
            DNMDTProductAuxVariable(),
            C,
            names,
            1:depth,
            time_steps;
            meta = meta * "_v",
        )
        wu_expr = add_expression_container!(
            container,
            DNMDTIntermediateSumExpression(),
            C,
            names,
            time_steps;
            meta = meta * "_wu",
        )
        wv_expr = add_expression_container!(
            container,
            DNMDTIntermediateSumExpression(),
            C,
            names,
            time_steps;
            meta = meta * "_wv",
        )
        bmc_cons = add_constraints_container!(
            container,
            DNMDTBinaryMcCormickConstraint(),
            C,
            names,
            1:depth,
            1:4,
            1:2, # u, v
            time_steps;
            sparse = true,
            meta,
        )
    else
        bmc_cons = add_constraints_container!(
            container,
            DNMDTBinaryMcCormickConstraint(),
            C,
            names,
            1:depth,
            1:4,
            time_steps;
            sparse = true,
            meta,
        )
    end
    
    dz_var = add_variable_container!(
        container,
        DNMDTResidualProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    zh_expr = add_expression_container!(
        container,
        DNMDTScaledProductExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    result_expr = add_expression_container!(
        container,
        BilinearProductExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    # ── Populate: binary expansions ──────────────────────────────────────

    xh_expr, beta_x_var, dx_var = _populate_binary_expansion!(
        container, C, names, time_steps,
        x_var, x_min, lx, depth, meta * "_x",
    )
    if double
        yh_expr, beta_y_var, dy_var = _populate_binary_expansion!(
            container, C, names, time_steps,
            y_var, y_min, ly, depth, meta * "_y",
        )
    else
        yh_expr = _scale!(
            container, C, names, time_steps,
            y_var, y_min, ly,
            meta * "_y"
        )
    end

    # ── Populate: blended terms, product aux, assembly, back-transform ───

    for name in names, t in time_steps
        # Blended terms (used locally in McCormick below)
        if double
            w_u = wu_expr[name, t] = JuMP.AffExpr(0.0)
            JuMP.add_to_expression!(w_u, lambda, dy_var[name, t])
            JuMP.add_to_expression!(w_u, 1 - lambda, yh_expr[name, t])

            w_v = wv_expr[name, t] = JuMP.AffExpr(0.0)
            JuMP.add_to_expression!(w_v, 1 - lambda, dx_var[name, t])
            JuMP.add_to_expression!(w_v, lambda, xh_expr[name, t])
        end

        # Binary McCormick for u[j] and v[j]
        for j in 1:depth
            u_j =
                u_var[name, j, t] = JuMP.@variable(
                    jump_model,
                    base_name = "DNMDTu_$(C)_{$(name), $(j), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = wu_hi,
                )
            if double
                v_j =
                    v_var[name, j, t] = JuMP.@variable(
                        jump_model,
                        base_name = "DNMDTv_$(C)_{$(name), $(j), $(t)}",
                        lower_bound = 0.0,
                        upper_bound = wv_hi,
                    )

                _add_mccormick_envelope!(
                    jump_model, bmc_cons, (name, j, 1, t),
                    w_u, beta_x_var[name, j, t], u_j,
                    0.0, wu_hi, 0.0, 1.0;
                )
                _add_mccormick_envelope!(
                    jump_model, bmc_cons, (name, j, 2, t),
                    w_v, beta_y_var[name, j, t], v_j,
                    0.0, wv_hi, 0.0, 1.0,
                )
            else
                _add_mccormick_envelope!(
                    jump_model, bmc_cons, (name, j, t),
                    yh_expr[name, t], beta_x_var[name, j, t], u_j,
                    0.0, 1.0, 0.0, 1.0,
                )
            end
        end

        # Residual product variable (bounded by [0, eps_L²])
        dz_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "DNMDTdz_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = eps_L * eps_L,
        )

        # Scaled product z_hat (used locally in back-transform below)
        zh = zh_expr[name, t] = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(zh, pow2_neg[j], u_var[name, j, t])
            if double
                JuMP.add_to_expression!(zh, pow2_neg[j], v_var[name, j, t])
            end
        end
        JuMP.add_to_expression!(zh, dz_var[name, t])

        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lx * ly, zh)
        JuMP.add_to_expression!(result, x_min * ly, yh_expr[name, t])
        JuMP.add_to_expression!(result, y_min * lx, xh_expr[name, t])
        JuMP.add_to_expression!(result, x_min * y_min)
    end

    # ── Residual McCormick (Delta_z ~ Delta_x * Delta_y) ──
    if double
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            dx_var, dy_var, dz_var,
            0.0, eps_L, 0.0, eps_L, meta * "_residual",
        )
    else
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            dx_var, yh_expr, dz_var,
            0.0, eps_L, 0.0, 1.0,
            meta * "_residual",
        )
    end

    # ── Global McCormick on z = x*y ──
    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, result_expr,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    return result_expr
end