# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses solver-native MOI.SOS2 constraints for adjacency enforcement.

struct QuadraticApproxVariable <: SparseVariableType end
struct QuadraticApproxLinkingConstraint <: ConstraintType end
struct QuadraticApproxNormalizationConstraint <: ConstraintType end

"""
    _add_sos2_quadratic_approx!(container, C, names, time_steps, x_var_container, x_min, x_max, num_segments)

Approximate x² using a piecewise linear function with solver-native SOS2 constraints.

Creates lambda (λ) variables representing convex combination weights over breakpoints,
adds linking, normalization, and MOI.SOS2 constraints, and returns a dictionary of JuMP
affine expressions approximating x².

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var_container`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `num_segments::Int`: number of PWL segments

# Returns
- `Dict{Tuple{String, Int}, JuMP.AffExpr}`: maps (name, t) to affine expression approximating x²
"""
function _add_sos2_quadratic_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    x_min::Float64,
    x_max::Float64,
    num_segments::Int,
) where {C <: IS.InfrastructureSystemsComponent}
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(x_min, x_max, _square; num_segments)
    n_points = num_segments + 1
    jump_model = get_jump_model(container)

    # Create all containers upfront
    lambda_container =
        add_variable_container!(container, QuadraticApproxVariable(), C)
    link_key = ConstraintKey(QuadraticApproxLinkingConstraint, C)
    _assign_container!(
        container.constraints,
        link_key,
        JuMP.Containers.SparseAxisArray(
            Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}(),
        ),
    )
    link_container =
        get_constraint(container, QuadraticApproxLinkingConstraint(), C)
    norm_key = ConstraintKey(QuadraticApproxNormalizationConstraint, C)
    _assign_container!(
        container.constraints,
        norm_key,
        JuMP.Containers.SparseAxisArray(
            Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}(),
        ),
    )
    norm_container =
        get_constraint(container, QuadraticApproxNormalizationConstraint(), C)

    result = Dict{Tuple{String, Int}, JuMP.AffExpr}()

    for name in names, t in time_steps
        x_var = x_var_container[name, t]

        # Create lambda variables: λ_i ∈ [0, 1]
        lambda = Vector{JuMP.VariableRef}(undef, n_points)
        for i in 1:n_points
            lambda[i] =
                lambda_container[(name, i, t)] = JuMP.@variable(
                    jump_model,
                    base_name = "QuadraticApproxVariable_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = 1.0,
                )
        end

        # x = Σ λ_i * x_i
        link_container[name, t] = JuMP.@constraint(
            jump_model,
            x_var == sum(lambda[i] * x_bkpts[i] for i in eachindex(x_bkpts))
        )

        # Σ λ_i = 1
        norm_container[name, t] = JuMP.@constraint(
            jump_model,
            sum(lambda) == 1.0
        )

        # λ ∈ SOS2 (solver-native)
        JuMP.@constraint(jump_model, lambda in MOI.SOS2(collect(1:n_points)))

        # Build x̂² = Σ λ_i * x_i² as an affine expression
        x_hat_sq = JuMP.AffExpr(0.0)
        for i in 1:n_points
            JuMP.add_to_expression!(x_hat_sq, x_sq_bkpts[i], lambda[i])
        end
        result[(name, t)] = x_hat_sq
    end

    return result
end

_square(x::Float64) = x * x
