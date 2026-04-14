# No-op bilinear approximation: returns exact x·y as a QuadExpr.
# For NLP-capable solvers or testing purposes.

"No-op bilinear config: returns exact x·y as a QuadExpr."
struct NoBilinearApproxConfig <: BilinearApproxConfig end

function _add_bilinear_approx!(
    ::NoBilinearApproxConfig,
    ::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    x_bounds::MinMax,
    y_bounds::MinMax,
    meta::String,
)
    return (result_expr = x * y,)
end

"""
    add_bilinear_approx!(::NoBilinearApproxConfig, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

No-op bilinear approximation: returns exact x·y as a QuadExpr.
"""
function add_bilinear_approx!(
    config::NoBilinearApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        BilinearProductExpression(),
        C,
        names,
        time_steps;
        meta,
        expr_type = JuMP.QuadExpr,
    )
    jump_model = get_jump_model(container)
    for (i, name) in enumerate(names), t in time_steps
        r = _add_bilinear_approx!(config, jump_model, x_var[name, t], y_var[name, t], x_bounds[i], y_bounds[i], meta)
        result_expr[name, t] = r.result_expr
    end
    return result_expr
end
