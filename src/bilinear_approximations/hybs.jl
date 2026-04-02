# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines Bin2 lower bound and Bin3 upper bound with shared sawtooth for x², y²
# and LP-only epigraph for (x+y)², (x−y)². Uses 2L binaries instead of 3L (Bin2).
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
Config for HybS (Hybrid Separable) bilinear approximation.
Combines Bin2 lower bound and Bin3 upper bound with shared quadratic for x², y²
and LP-only epigraph for (x+y)², (x−y)². Epigraph depth is computed as max(depth, 1).
"""
struct HybSConfig <: BilinearApproxConfig
    quad_config::QuadraticApproxConfig
    epigraph_depth::Int
    add_mccormick::Bool
end
HybSConfig(quad_config::QuadraticApproxConfig, epigraph_depth::Int) =
    HybSConfig(quad_config, epigraph_depth, true)

# --- Unified HybS dispatch methods ---

"""
    _add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta)

Approximate x·y using HybS (Hybrid Separable) relaxation with config-selected quadratic method.
"""
function _add_bilinear_approx!(
    config::HybSConfig,
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
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    xsq = _add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        x_var, x_min, x_max, meta * "_x",
    )
    ysq = _add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        y_var, y_min, y_max, meta * "_y",
    )
    return _add_bilinear_approx!(
        config, container, C, names, time_steps,
        xsq, ysq, x_var, y_var,
        x_min, x_max, y_min, y_max, meta,
    )
end

"""
    _add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, xsq, ysq, depth, meta)

HybS bilinear approximation with pre-computed quadratic approximations for x² and y².

Combines Bin2 and Bin3 separable identities:
- Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y) where z_p1 lower-bounds (x+y)²
- Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2) where z_p2 lower-bounds (x−y)²

The cross-terms (x+y)² and (x−y)² always use epigraph Q^{L1} (pure LP).
"""
function _add_bilinear_approx!(
    config::HybSConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    xsq,
    ysq,
    x_var,
    y_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    # Bounds for auxiliary variables
    p1_min = x_min + y_min
    p1_max = x_max + y_max
    p2_min = x_min - y_max     # p2 = x − y, so min uses −y_max
    p2_max = x_max - y_min     # and max uses −y_min
    IS.@assert_op x_max > x_min
    IS.@assert_op y_max > y_min

    jump_model = get_jump_model(container)

    # Meta suffixes for cross-term expressions
    meta_p1 = meta * "_plus"
    meta_p2 = meta * "_diff"

    p1_expr = add_expression_container!(
        container,
        VariableSumExpression(),
        C,
        names,
        time_steps;
        meta = meta_p1,
    )
    p2_expr = add_expression_container!(
        container,
        VariableDifferenceExpression(),
        C,
        names,
        time_steps;
        meta = meta_p2,
    )

    for name in names, t in time_steps
        x = x_var[name, t]
        y = y_var[name, t]

        # p1 = x + y
        p1 = p1_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p1, x, 1.0)
        add_proportional_to_jump_expression!(p1, y, 1.0)

        # p2 = x − y
        p2 = p2_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p2, x, 1.0)
        add_proportional_to_jump_expression!(p2, y, -1.0)
    end

    # --- Epigraph Q^{L1} lower bound for (x+y)² and (x−y)² (no binaries) ---
    epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
    zp1_expr = _add_quadratic_approx!(
        epi_cfg,
        container, C, names, time_steps,
        p1_expr, p1_min, p1_max, meta_p1,
    )
    zp2_expr = _add_quadratic_approx!(
        epi_cfg,
        container, C, names, time_steps,
        p2_expr, p2_min, p2_max, meta_p2,
    )

    # --- Create z variable and two-sided HybS bounds ---
    z_var = add_variable_container!(
        container,
        BilinearProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    hybrid_cons = add_constraints_container!(
        container,
        HybSBoundConstraint(),
        C,
        names,
        1:2,
        time_steps;
        sparse = true,
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

    # Compute valid bounds for z ≈ x·y from variable bounds
    z_lo = min(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    z_hi = max(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)

    for name in names, t in time_steps
        z =
            z_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "HybSProduct_$(C)_{$(name), $(t)}",
                lower_bound = z_lo,
                upper_bound = z_hi,
            )

        zx = xsq[name, t]
        zy = ysq[name, t]
        zp1 = zp1_expr[name, t]
        zp2 = zp2_expr[name, t]

        # Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y)
        hybrid_cons[(name, 1, t)] = JuMP.@constraint(
            jump_model,
            z >= 0.5 * (zp1 - zx - zy),
        )
        # Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2)
        hybrid_cons[(name, 2, t)] = JuMP.@constraint(
            jump_model,
            z <= 0.5 * (zx + zy - zp2),
        )

        result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
    end

    # --- Reformulated McCormick cuts through separable variables ---
    if config.add_mccormick
        _add_reformulated_mccormick!(
            container, C, names, time_steps,
            x_var, y_var, zp1_expr, zp2_expr, xsq, ysq,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    return result_expr
end
