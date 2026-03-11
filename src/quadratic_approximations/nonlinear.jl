struct BilinearProductExpression <: ExpressionType end

function _add_bilinear!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    y_var_container,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    z_container = add_expression_container!(
        container,
        BilinearProductExpression(),
        C,
        names,
        time_steps;
        expr_type = JuMP.QuadExpr,
        meta,
    )
    for name in names, t in time_steps
        z_expr = JuMP.QuadExpr()
        JuMP.add_to_expression!(z_expr, x_var_container[name, t], y_var_container[name, t])
        z_container[name, t] = z_expr
    end
    return z_container
end
