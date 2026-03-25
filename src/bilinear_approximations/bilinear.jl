# Bin2 separable approximation of bilinear products z = x·y.
# Uses the identity: x·y = (1/2)*((x+y)² − x² - y²).
# Calls existing quadratic approximation functions for p²=(x+y)²

"Expression container for bilinear product (x·y) approximation results."
struct BilinearProductExpression <: ExpressionType end
"Variable container for bilinear product (x ̇y) approximation results."
struct BilinearProductVariable <: VariableType end
"Expression container for adding variables."
struct VariableSumExpression <: ExpressionType end
"Expression container for subtracting variables."
struct VariableDifferenceExpression <: ExpressionType end
"Constraint container for linking product expressions and variables."
struct BilinearProductLinkingConstraint <: ConstraintType end

"""
    _add_bilinear_approx_impl!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, zx_expr, zy_expr, quad_approx_fn, refinement, meta; quad_kwargs...)

Internal implementation for Bin2 bilinear approximation using z = (1/2)((x+y)² − x² - y²).

Accepts pre-computed quadratic approximations of x² and y², creates auxiliary
variables p = x+y, calls `quad_approx_fn` to approximate p², then combines via
the multiplicative identity. Stores affine expressions approximating x·y in a
`BilinearProductExpression` expression container.

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
- `quad_approx_fn`: quadratic approximation function with signature
  `(container, C, names, ts, var, lo, hi, refinement, meta; kwargs...) → expr_container`
- `refinement`: refinement parameter forwarded to `quad_approx_fn` (e.g., num_segments or depth)
- `meta::String`: identifier for container keys
- `quad_kwargs...`: keyword arguments forwarded to `quad_approx_fn`
"""
function _add_bilinear_approx_impl!(
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
    quad_approx_fn,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    # Bounds for p = x + y
    p_min = x_min + y_min
    p_max = x_max + y_max
    IS.@assert_op p_min <= p_max

    jump_model = get_jump_model(container)
    meta_plus = meta * "_plus"

    p_expr = add_expression_container!(
        container,
        VariableSumExpression(),
        C,
        names,
        time_steps;
        meta = meta_plus,
    )
    for name in names, t in time_steps
        p = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p, x_var[name, t], 1.0)
        add_proportional_to_jump_expression!(p, y_var[name, t], 1.0)
        p_expr[name, t] = p
    end

    # Approximate p² = (x+y)² using the provided quadratic approximation function
    zp_expr = quad_approx_fn(p_expr, p_min, p_max, meta_plus)

    z_var = add_variable_container!(
        container,
        BilinearProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        BilinearProductLinkingConstraint(),
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

    # Compute valid bounds for z = x·y from variable bounds
    z_lo = min(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    z_hi = max(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)

    for name in names, t in time_steps
        # It's not necessary to create a variable container here, but it is
        # necessary in HybS, so this is here for symmetry.
        z =
            z_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "BilinearProduct_$(C)_{$(name), $(t)}",
                lower_bound = z_lo,
                upper_bound = z_hi,
            )

        # z = (1/2) * (p² − x² - y²)
        z_expr = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(z_expr, zp_expr[name, t], 0.5)
        add_proportional_to_jump_expression!(z_expr, zx_expr[name, t], -0.5)
        add_proportional_to_jump_expression!(z_expr, zy_expr[name, t], -0.5)
        link_cons[name, t] = JuMP.@constraint(jump_model, z == z_expr)

        result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
    end

    return result_expr
end

"""
    _add_sos2_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, num_segments, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with solver-native SOS2 quadratic approximations.

# Arguments
Same as `_add_bilinear_approx_impl!` plus:
- `num_segments::Int`: number of PWL segments for each quadratic approximation
"""
function _add_sos2_bilinear_approx!(
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
    num_segments::Int,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    zx_expr = _add_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, num_segments, meta_x;
        add_mccormick,
    )
    zy_expr = _add_sos2_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, num_segments, meta_y;
        add_mccormick,
    )
    return _add_sos2_bilinear_approx!(
        container, C, names, time_steps,
        x_var, y_var, x_min, x_max, y_min, y_max,
        zx_expr, zy_expr, num_segments, meta;
        add_mccormick,
    )
end

"""
    _add_sos2_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, zx_precomputed, zy_precomputed, num_segments, meta; add_mccormick)

Approximate x·y using Bin2+SOS2 with pre-computed quadratic approximations for x² and y².
"""
function _add_sos2_bilinear_approx!(
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
    num_segments::Int,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    qa = (x, lo, hi, meta) -> _add_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x, lo, hi,
        num_segments, meta; add_mccormick
    )
    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_precomputed, zy_precomputed, qa, meta;
    )
end

"""
    _add_manual_sos2_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, num_segments, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with manual SOS2 quadratic approximations.

# Arguments
Same as `_add_bilinear_approx_impl!` plus:
- `num_segments::Int`: number of PWL segments for each quadratic approximation
"""
function _add_manual_sos2_bilinear_approx!(
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
    num_segments::Int,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    zx_expr = _add_manual_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, num_segments, meta_x;
        add_mccormick,
    )
    zy_expr = _add_manual_sos2_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, num_segments, meta_y;
        add_mccormick,
    )
    return _add_manual_sos2_bilinear_approx!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_expr, zy_expr, num_segments, meta;
        add_mccormick,
    )
end

"""
    _add_manual_sos2_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, zx_precomputed, zy_precomputed, num_segments, meta; add_mccormick)

Approximate x·y using Bin2+ManualSOS2 with pre-computed quadratic approximations for x² and y².
"""
function _add_manual_sos2_bilinear_approx!(
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
    num_segments::Int,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    qa = (x, lo, hi, meta) -> _add_manual_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x, lo, hi,
        num_segments, meta; add_mccormick
    )
    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_precomputed, zy_precomputed, qa, meta;
    )
end

"""
    _add_sawtooth_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with sawtooth quadratic approximations.

# Arguments
Same as `_add_bilinear_approx_impl!` plus:
- `depth::Int`: sawtooth depth (number of binary variables per quadratic approximation)
"""
function _add_sawtooth_bilinear_approx!(
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
    tighten::Bool = false,
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    zx_expr = _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta_x;
        tighten, add_mccormick,
    )
    zy_expr = _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta_y;
        tighten, add_mccormick,
    )
    return _add_sawtooth_bilinear_approx!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_expr, zy_expr, depth, meta;
        tighten, add_mccormick,
    )
end

"""
    _add_sawtooth_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, zx_precomputed, zy_precomputed, depth, meta; tighten, add_mccormick)

Approximate x·y using Bin2+Sawtooth with pre-computed quadratic approximations for x² and y².
"""
function _add_sawtooth_bilinear_approx!(
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
    tighten::Bool = false,
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    qa = (x, lo, hi, meta) -> _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        x, lo, hi,
        depth, meta; tighten, add_mccormick
    )
    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_precomputed, zy_precomputed, qa, meta;
    )
end

function _add_dnmdt_quadratic_bilinear_approx!(
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
    double::Bool = false,
    tighten::Bool = false,
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    zx_expr = quad_fn(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta_x,
    )
    zy_expr = quad_fn(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta_y,
    )
    return _add_dnmdt_quadratic_bilinear_approx!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max,
        zx_expr, zy_expr, depth, meta;
        double, tighten, add_mccormick
    )
end

function _add_dnmdt_quadratic_bilinear_approx!(
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
    double::Bool = false,
    tighten::Bool = false,
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    qa = if double
        (x, lo, hi, meta) -> _add_dnmdt_quadratic_approx!(
            container, C, names, time_steps,
            x, lo, hi,
            depth, meta; tighten
        )
    else
        (x, lo, hi, meta) -> _add_nmdt_quadratic_approx!(
            container, C, names, time_steps,
            x, lo, hi,
            depth, meta; tighten
        ) 
    end

    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var, x_min, x_max, y_min, y_max,
        zx_precomputed, zy_precomputed, qa, meta;
    )
end
