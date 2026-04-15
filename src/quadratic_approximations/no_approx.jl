# No-op quadratic approximation: returns exact x² as a QuadExpr.
# For NLP-capable solvers or testing purposes.

"No-op config: returns exact x² as a QuadExpr (for NLP-capable solvers or testing)."
struct NoQuadApproxConfig <: QuadraticApproxConfig end

"""
    _add_quadratic_approx!(::NoQuadApproxConfig, container, C, names, time_steps, x_var, x_min, x_max, meta)

No-op quadratic approximation: returns exact x² as a QuadExpr.

# Arguments
- `::NoQuadApproxConfig`: no-op configuration (no fields)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `meta::String`: variable type identifier for the approximation
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
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        QuadraticExpression,
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
