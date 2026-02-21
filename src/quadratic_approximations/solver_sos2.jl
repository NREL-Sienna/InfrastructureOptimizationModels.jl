# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses solver-native MOI.SOS2 constraints for adjacency enforcement.

struct QuadraticApproxVariable <: SparseVariableType end
struct QuadraticApproxLinkingConstraint <: ConstraintType end
struct QuadraticApproxNormalizationConstraint <: ConstraintType end

"""
    _add_sos2_quadratic_approx!(container, C, name, t, x_var, x_min, x_max, num_segments)

Approximate x² using a piecewise linear function with solver-native SOS2 constraints.

Creates lambda (λ) variables representing convex combination weights over breakpoints,
adds linking, normalization, and MOI.SOS2 constraints, and returns a JuMP affine
expression approximating x².

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `name::String`: component name
- `t::Int`: time period
- `x_var::JuMP.VariableRef`: variable whose square is being approximated
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `num_segments::Int`: number of PWL segments

# Returns
- `JuMP.AffExpr`: affine expression approximating x²
"""
function _add_sos2_quadratic_approx!(
    container::OptimizationContainer,
    ::Type{C},
    name::String,
    t::Int,
    x_var::JuMP.VariableRef,
    x_min::Float64,
    x_max::Float64,
    num_segments::Int,
) where {C <: IS.InfrastructureSystemsComponent}
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(x_min, x_max, _square; num_segments)
    n_points = num_segments + 1

    # Create lambda variables: λ_i ∈ [0, 1]
    lambda = add_pwl_variables!(container, QuadraticApproxVariable, C, name, t, n_points)

    # x = Σ λ_i * x_i
    add_pwl_linking_constraint!(
        container,
        QuadraticApproxLinkingConstraint,
        C,
        name,
        t,
        x_var,
        lambda,
        x_bkpts,
    )

    # Σ λ_i = 1
    add_pwl_normalization_constraint!(
        container,
        QuadraticApproxNormalizationConstraint,
        C,
        name,
        t,
        lambda,
        1.0,
    )

    # λ ∈ SOS2 (solver-native)
    add_pwl_sos2_constraint!(container, C, name, t, lambda)

    # Build x̂² = Σ λ_i * x_i² as an affine expression
    x_hat_sq = JuMP.AffExpr(0.0)
    for i in 1:n_points
        JuMP.add_to_expression!(x_hat_sq, x_sq_bkpts[i], lambda[i])
    end
    return x_hat_sq
end

_square(x::Float64) = x * x
