"""
NLP methods with the same interface as bilinear and quadratic approximations
for use in building formulations.
"""

function _add_bilinear_nlp!(
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
    refinement::Int,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        IOM.BilinearProductExpression(),
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

function _add_quadratic_nlp!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    refinement::Int,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        IOM.QuadraticExpression(),
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
