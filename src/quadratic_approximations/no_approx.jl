# No-op quadratic approximation: returns exact x² as a QuadExpr.
# For NLP-capable solvers or testing purposes.

"No-op config: returns exact x² as a QuadExpr (for NLP-capable solvers or testing)."
struct NoQuadApproxConfig <: QuadraticApproxConfig end

"""
    _add_quadratic_approx!(::NoQuadApproxConfig, container, C, names, time_steps, x_var, x_min, x_max, depth, meta)

No-op quadratic approximation: returns exact x² as a QuadExpr.
The `depth` parameter is ignored.
"""
function _add_quadratic_approx!(
    ::NoQuadApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        QuadraticExpression(),
        C,
        names,
        time_steps;
        meta,
        expr_type = JuMP.QuadExpr,
    )
    for name in names, t in time_steps
        result_expr[name, t] = x_var[name, t] * x_var[name, t]
    end
    return result_expr
end
