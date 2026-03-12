# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses manually-implemented SOS2 adjacency via binary variables and linear constraints.

"Binary segment-selection variables (z) for manual SOS2 quadratic approximation."
struct ManualSOS2BinaryVariable <: SparseVariableType end
"Ensures exactly one segment is active (∑zⱼ = 1) in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionConstraint <: ConstraintType end
"Links active segment to lambda variables."
struct ManualSOS2AdjacencyConstraint <: ConstraintType end

"""
    _add_manual_sos2_quadratic_approx!(container, C, names, time_steps, x_var, x_min, x_max, num_segments, meta)

Approximate x² using a piecewise linear function with manually-implemented SOS2 constraints.

Creates lambda (λ) variables representing convex combination weights over breakpoints,
adds linking, normalization, and manual adjacency constraints using binary variables,
and stores affine expressions approximating x² in a `QuadraticApproxExpression`
expression container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `num_segments::Int`: number of PWL segments
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function _add_manual_sos2_quadratic_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    num_segments::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(x_min, x_max, _square; num_segments)
    n_points = num_segments + 1
    n_bins = n_points - 1
    jump_model = get_jump_model(container)

    # Create all containers upfront
    lambda_container =
        add_variable_container!(container, QuadraticApproxVariable(), C; meta)
    z_container = add_variable_container!(container, ManualSOS2BinaryVariable(), C; meta)
    link_cons = @_add_container(constraints, SOS2LinkingConstraint)
    norm_cons = @_add_container(constraints, SOS2NormConstraint)
    seg_cons = @_add_container(constraints, ManualSOS2SegmentSelectionConstraint)
    adj_cons = @_add_container(constraints, ManualSOS2AdjacencyConstraint, 1:n_points)
    result_expr = @_add_container(expression, QuadraticApproxExpression)

    for name in names, t in time_steps
        x = x_var[name, t]

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
        link_cons[name, t] = JuMP.@constraint(
            jump_model,
            x == sum(lambda[i] * x_bkpts[i] for i in eachindex(x_bkpts))
        )

        # Σ λ_i = 1
        norm_cons[name, t] = JuMP.@constraint(jump_model, sum(lambda) == 1.0)

        # Create binary segment-selection variables z_j
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
        seg_cons[name, t] = JuMP.@constraint(jump_model, sum(z_vars) == 1)

        # Adjacency constraints: λ_i ≤ z_{i-1} + z_i (with boundary cases)
        # λ_1 ≤ z_1
        adj_cons[name, 1, t] = JuMP.@constraint(jump_model, lambda[1] <= z_vars[1])
        # λ_i ≤ z_{i-1} + z_i for i = 2..n-1
        for i in 2:(n_points - 1)
            adj_cons[name, i + 1, t] =
                JuMP.@constraint(jump_model, lambda[i] <= z_vars[i - 1] + z_vars[i])
        end
        # λ_n ≤ z_{n-1}
        adj_cons[name, n_points, t] =
            JuMP.@constraint(jump_model, lambda[n_points] <= z_vars[n_bins])

        # Build x̂² = Σ λ_i * x_i² as an affine expression
        x_hat_sq = JuMP.AffExpr(0.0)
        for i in 1:n_points
            JuMP.add_to_expression!(x_hat_sq, x_sq_bkpts[i], lambda[i])
        end
        result_expr[name, t] = x_hat_sq
    end

    return result_expr
end
