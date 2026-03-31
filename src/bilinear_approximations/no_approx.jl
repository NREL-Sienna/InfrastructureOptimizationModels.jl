# No-op bilinear approximation: returns exact x·y as a QuadExpr.
# For NLP-capable solvers or testing purposes.

"No-op bilinear config: returns exact x·y as a QuadExpr."
struct NoBilinearApproxConfig <: BilinearApproxConfig end

"""
    _add_bilinear_approx!(::NoBilinearApproxConfig, container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta)

No-op bilinear approximation: returns exact x·y as a QuadExpr.
The `depth` parameter is ignored.
"""
function _add_bilinear_approx!(
    ::NoBilinearApproxConfig,
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
    for name in names, t in time_steps
        result_expr[name, t] = x_var[name, t] * y_var[name, t]
    end
    return result_expr
end
