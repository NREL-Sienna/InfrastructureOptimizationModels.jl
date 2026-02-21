# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses manually-implemented SOS2 adjacency via binary variables and linear constraints.

struct ManualSOS2BinaryVariable <: SparseVariableType end
struct ManualSOS2SegmentSelectionConstraint <: ConstraintType end

"""
    _add_manual_sos2_adjacency_constraints!(container, C, name, t, lambda)

Enforce the SOS2 adjacency condition on lambda variables using binary segment-selection
variables and linear constraints.

Creates n-1 binary variables z_j and adds:
- Σ z_j = 1 (exactly one segment active)
- λ_1 ≤ z_1
- λ_i ≤ z_{i-1} + z_i for i = 2..n-1
- λ_n ≤ z_{n-1}
"""
function _add_manual_sos2_adjacency_constraints!(
    container::OptimizationContainer,
    ::Type{C},
    name::String,
    t::Int,
    lambda::Vector{JuMP.VariableRef},
) where {C <: IS.InfrastructureSystemsComponent}
    n = length(lambda)
    n_bins = n - 1
    jump_model = get_jump_model(container)

    # Create binary segment-selection variables z_j
    z_container = lazy_container_addition!(container, ManualSOS2BinaryVariable(), C)
    z_vars = Vector{JuMP.VariableRef}(undef, n_bins)
    for j in 1:n_bins
        z_vars[j] =
            z_container[(name, j, t)] = JuMP.@variable(
                jump_model,
                base_name = "ManualSOS2Binary_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
    end

    # Σ z_j = 1 (segment selection)
    if !has_container_key(container, ManualSOS2SegmentSelectionConstraint, C)
        con_key = ConstraintKey(ManualSOS2SegmentSelectionConstraint, C)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}()
        _assign_container!(
            container.constraints,
            con_key,
            JuMP.Containers.SparseAxisArray(contents),
        )
    end
    seg_container =
        get_constraint(container, ManualSOS2SegmentSelectionConstraint(), C)
    seg_container[name, t] = JuMP.@constraint(jump_model, sum(z_vars) == 1)

    # Adjacency constraints: λ_i ≤ z_{i-1} + z_i (with boundary cases)
    # λ_1 ≤ z_1
    JuMP.@constraint(jump_model, lambda[1] <= z_vars[1])
    # λ_i ≤ z_{i-1} + z_i for i = 2..n-1
    for i in 2:(n - 1)
        JuMP.@constraint(jump_model, lambda[i] <= z_vars[i - 1] + z_vars[i])
    end
    # λ_n ≤ z_{n-1}
    JuMP.@constraint(jump_model, lambda[n] <= z_vars[n_bins])
    return
end

"""
    _add_manual_sos2_quadratic_approx!(container, C, name, t, x_var, x_min, x_max, num_segments)

Approximate x² using a piecewise linear function with manually-implemented SOS2 constraints.

Creates lambda (λ) variables representing convex combination weights over breakpoints,
adds linking, normalization, and manual adjacency constraints using binary variables,
and returns a JuMP affine expression approximating x².

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
function _add_manual_sos2_quadratic_approx!(
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

    # Manual SOS2 adjacency via binary variables
    _add_manual_sos2_adjacency_constraints!(container, C, name, t, lambda)

    # Build x̂² = Σ λ_i * x_i² as an affine expression
    x_hat_sq = JuMP.AffExpr(0.0)
    for i in 1:n_points
        JuMP.add_to_expression!(x_hat_sq, x_sq_bkpts[i], lambda[i])
    end
    return x_hat_sq
end
