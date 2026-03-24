"Expression container for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1]."
struct NormedVariableExpression <: ExpressionType end

"""
    _normed_variable!(container, C, names, time_steps, x_var, x_min, x_max, meta)

Create an affine expression for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1].

Stores results in a `NormedVariableExpression` expression container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _normed_variable!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    lx = x_max - x_min
    result_expr = add_expression_container!(
        container,
        NormedVariableExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        add_linear_to_jump_expression!(result, x_var[name, t], 1.0 / lx, -x_min / lx)
    end
    return result_expr
end
