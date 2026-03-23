# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines Bin2 lower bound and Bin3 upper bound with shared sawtooth for x², y²
# and LP-only epigraph for (x+y)², (x−y)². Uses 2L binaries instead of 3L (Bin2).
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
    _add_hybs_bilinear_approx_impl!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, zx_expr, zy_expr, epigraph_depth, meta; add_mccormick)

Approximate x·y using the HybS (Hybrid Separable) relaxation from Beach et al. (2024).

Accepts pre-computed quadratic approximations of x² and y². Combines Bin2 and Bin3
separable identities:
- Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y) where z_p1 lower-bounds (x+y)²
- Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2) where z_p2 lower-bounds (x−y)²

The cross-terms (x+y)² and (x−y)² always use epigraph Q^{L1} (pure LP).

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `zx_expr`: pre-computed quadratic approximation of x², indexed by (name, t)
- `zy_expr`: pre-computed quadratic approximation of y², indexed by (name, t)
- `epigraph_depth::Int`: depth for epigraph approximation of (x+y)² and (x−y)²
- `meta::String`: identifier encoding the original variable type being approximated
- `add_mccormick::Bool`: whether to add McCormick envelope constraints (default: false)
"""
function _add_hybs_bilinear_approx_impl!(
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
    zx_expr,
    zy_expr,
    epigraph_depth::Int,
    meta::String;
    add_mccormick::Bool = false,
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
        JuMP.add_to_expression!(p1, x)
        JuMP.add_to_expression!(p1, y)

        # p2 = x − y
        p2 = p2_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(p2, x)
        JuMP.add_to_expression!(p2, -1.0, y)
    end

    # --- Epigraph Q^{L1} lower bound for (x+y)² and (x−y)² (no binaries) ---
    zp1_expr = _add_epigraph_quadratic_approx!(
        container, C, names, time_steps,
        p1_expr, p1_min, p1_max, epigraph_depth, meta_p1,
    )
    zp2_expr = _add_epigraph_quadratic_approx!(
        container, C, names, time_steps,
        p2_expr, p2_min, p2_max, epigraph_depth, meta_p2,
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

        zx = zx_expr[name, t]
        zy = zy_expr[name, t]
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

    # --- Step 6: McCormick envelope for additional tightening ---
    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, z_var,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    return result_expr
end

function _add_hybs_sos2_bilinear_approx!(
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
    add_mccormick::Bool = false,
    add_quad_mccormick::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth)),
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    zx_expr = _add_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta_x;
        add_mccormick = add_quad_mccormick,
    )
    zy_expr = _add_sos2_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta_y;
        add_mccormick = add_quad_mccormick,
    )
    return _add_hybs_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_expr, zy_expr, epigraph_depth, meta;
        add_mccormick,
    )
end

function _add_hybs_sos2_bilinear_approx!(
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
    zx_precomputed,
    zy_precomputed,
    depth::Int,
    meta::String;
    add_mccormick::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth)),
) where {C <: IS.InfrastructureSystemsComponent}
    return _add_hybs_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_precomputed, zy_precomputed, epigraph_depth, meta;
        add_mccormick,
    )
end

function _add_hybs_manual_sos2_bilinear_approx!(
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
    add_mccormick::Bool = false,
    add_quad_mccormick::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth)),
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    zx_expr = _add_manual_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta_x;
        add_mccormick = add_quad_mccormick,
    )
    zy_expr = _add_manual_sos2_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta_y;
        add_mccormick = add_quad_mccormick,
    )
    return _add_hybs_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_expr, zy_expr, epigraph_depth, meta;
        add_mccormick,
    )
end

function _add_hybs_manual_sos2_bilinear_approx!(
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
    zx_precomputed,
    zy_precomputed,
    depth::Int,
    meta::String;
    add_mccormick::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth)),
) where {C <: IS.InfrastructureSystemsComponent}
    return _add_hybs_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_precomputed, zy_precomputed, epigraph_depth, meta;
        add_mccormick,
    )
end

function _add_hybs_sawtooth_bilinear_approx!(
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
    add_mccormick::Bool = false,
    add_quad_mccormick::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth)),
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    zx_expr = _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta_x;
        tighten, add_mccormick = add_quad_mccormick,
    )
    zy_expr = _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta_y;
        tighten, add_mccormick = add_quad_mccormick,
    )
    return _add_hybs_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_expr, zy_expr, epigraph_depth, meta;
        add_mccormick,
    )
end

function _add_hybs_sawtooth_bilinear_approx!(
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
    zx_precomputed,
    zy_precomputed,
    depth::Int,
    meta::String;
    add_mccormick::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth)),
) where {C <: IS.InfrastructureSystemsComponent}
    return _add_hybs_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_precomputed, zy_precomputed, epigraph_depth, meta;
        add_mccormick,
    )
end
