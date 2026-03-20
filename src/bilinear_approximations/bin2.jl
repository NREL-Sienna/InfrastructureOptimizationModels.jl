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
    _add_bin2_bilinear_approx_impl!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, quad_approx_fn, meta)

Internal implementation for Bin2 bilinear approximation using z = (1/2)((x+y)² − x² - y²).

Creates auxiliary variables p = x+y, calls `quad_approx_fn` to
approximate p², then combines via multiplicative identity. Stores affine expressions
approximating x·y in a `BilinearProductExpression` expression container.

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
- `quad_approx_fn`: callable with signature (container, C, names, ts, var_cont, lo, hi, meta) → nothing
- `meta::String`: identifier for container keys
"""
function _add_bin2_bilinear_approx_impl!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    xsq_expr,
    ysq_expr,
    quad_approx_fn!,
    depth,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    # Bounds for p = x + y
    p_min = x_min + y_min
    p_max = x_max + y_max
    IS.@assert_op p_min <= p_max

    jump_model = get_jump_model(container)

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
        JuMP.add_to_expression!(p, x_var[name, t])
        JuMP.add_to_expression!(p, y_var[name, t])
        p_expr[name, t] = p
    end

    # Approximate p², x², y² using the provided quadratic approximation function
    psq_expr = quad_approx_fn!(
        container, C, names, time_steps,
        p_expr, p_min, p_max, depth,
        meta * "_plus",
    )

    result_expr = add_expression_container!(
        container,
        BilinearProductExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        # z = (1/2) * (p² − x² - y²)
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, 0.5, psq_expr[name, t])
        JuMP.add_to_expression!(result, -0.5, xsq_expr[name, t])
        JuMP.add_to_expression!(result, -0.5, ysq_expr[name, t])
    end

    return result_expr
end

"""
    _add_bin2_sos2_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with solver-native SOS2 quadratic approximations.

# Arguments
Same as `_add_bin2_bilinear_approx_impl!` plus:
- `depth::Int`: number of PWL segments for each quadratic approximation
"""
function _add_bin2_sos2_bilinear_approx!(
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
) where {C <: IS.InfrastructureSystemsComponent}
    xsq_expr = _add_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth,
        meta * "_x"; add_mccormick,
    )
    ysq_expr = _add_sos2_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth,
        meta * "_x"; add_mccormick,
    )
    return _add_bin2_bilinear_approx_impl!(
        container, C, names, time_steps,
        xsq_expr, ysq_expr, _add_sos2_quadratic_approx!,
        depth, meta,
    )
end

"""
    _add_bin2_manual_sos2_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with manual SOS2 quadratic approximations.

# Arguments
Same as `_add_bin2_bilinear_approx_impl!` plus:
- `depth::Int`: number of PWL segments for each quadratic approximation
"""
function _add_bin2_manual_sos2_bilinear_approx!(
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
) where {C <: IS.InfrastructureSystemsComponent}
    xsq_expr = _add_manual_sos2_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth,
        meta * "_x"; add_mccormick,
    )
    ysq_expr = _add_manual_sos2_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth,
        meta * "_x"; add_mccormick,
    )
    return _add_bin2_bilinear_approx_impl!(
        container, C, names, time_steps,
        xsq_expr, ysq_expr, _add_manual_sos2_quadratic_approx!,
        depth, meta,
    )
end

"""
    _add_bin2_sawtooth_bilinear_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with sawtooth quadratic approximations.

# Arguments
Same as `_add_bin2_bilinear_approx_impl!` plus:
- `depth::Int`: sawtooth depth (number of binary variables per quadratic approximation)
"""
function _add_bin2_sawtooth_bilinear_approx!(
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
    xsq_expr = _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth,
        meta * "_x"; tighten,
        add_mccormick,
    )
    ysq_expr = _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth,
        meta * "_x"; tighten,
        add_mccormick,
    )
    quad_fn = (args...) -> _add_sawtooth_quadratic_approx!(args...; tighten)
    return _add_bin2_bilinear_approx_impl!(
        container, C, names, time_steps,
        xsq_expr, ysq_expr, quad_fn,
        depth, meta,
    )
end

function _add_bin2_dnmdt_bilinear_approx!(
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
    quad_fn =
        (cont, CT, nms, ts, vc, lo, hi, m) ->
            _add_dnmdt_univariate_approx!(
                cont,
                CT,
                nms,
                ts,
                vc,
                lo,
                hi,
                depth,
                m;
                double,
                tighten,
                add_mccormick,
            )
    return _add_bin2_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var, y_var,
        x_min, x_max, y_min, y_max, quad_fn, meta;
    )
end
