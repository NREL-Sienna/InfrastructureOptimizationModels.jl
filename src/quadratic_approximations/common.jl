"Expression container for the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1]."
struct NormedVariableExpression <: ExpressionType end

"Expression container for quadratic (x²) approximation results."
struct QuadraticExpression <: ExpressionType end

# --- Quadratic approximation config hierarchy ---

"Abstract supertype for quadratic approximation method configurations."
abstract type QuadraticApproxConfig end

"""
    _normed_variable(x, bounds)

Compute the normalized variable xh = (x − x_min) / (x_max − x_min) ∈ [0,1].

Returns a `JuMP.AffExpr`. This is a pure inner helper — container-unaware.

# Arguments
- `x::JuMP.AbstractJuMPScalar`: the variable to normalize
- `bounds::MinMax`: `(min = x_min, max = x_max)` bounds for this name
"""
function _normed_variable(
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
)
    IS.@assert_op bounds.max > bounds.min
    lx = bounds.max - bounds.min
    result = JuMP.AffExpr(0.0)
    add_linear_to_jump_expression!(result, x, 1.0 / lx, -bounds.min / lx)
    return result
end
