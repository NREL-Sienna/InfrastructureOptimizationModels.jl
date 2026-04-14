# No-op quadratic approximation: returns exact x² as a QuadExpr.
# For NLP-capable solvers or testing purposes.

"No-op config: returns exact x² as a QuadExpr (for NLP-capable solvers or testing)."
struct NoQuadApproxConfig <: QuadraticApproxConfig end

"""
    _add_quadratic_approx!(::NoQuadApproxConfig, model, x, bounds, meta)

Inner no-op quadratic approximation: returns exact x² as a QuadExpr.

# Arguments
- `::NoQuadApproxConfig`: no-op configuration (no fields)
- `::JuMP.Model`: unused but kept for uniform interface
- `x::JuMP.AbstractJuMPScalar`: the variable to square
- `bounds::MinMax`: unused but kept for uniform interface
- `meta::String`: unused but kept for uniform interface
"""
function _add_quadratic_approx!(
    ::NoQuadApproxConfig,
    ::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    meta::String,
)
    return (result_expr = x * x,)
end

"""
    add_quadratic_approx!(::NoQuadApproxConfig, container, C, names, time_steps, x_var, bounds, meta)

No-op quadratic approximation: returns exact x² as a QuadExpr.

# Arguments
- `::NoQuadApproxConfig`: no-op configuration (no fields)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name bounds [(min=x_min, max=x_max), ...]
- `meta::String`: variable type identifier for the approximation
"""
function add_quadratic_approx!(
    config::NoQuadApproxConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
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
    jump_model = get_jump_model(container)
    for (i, name) in enumerate(names), t in time_steps
        r = _add_quadratic_approx!(config, jump_model, x_var[name, t], bounds[i], meta)
        result_expr[name, t] = r.result_expr
    end
    return result_expr
end
