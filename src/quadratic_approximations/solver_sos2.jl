# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses solver-native MOI.SOS2 constraints for adjacency enforcement.

"lambda_var (λ) convex combination weight variables for SOS2 quadratic approximation."
struct QuadraticVariable <: SparseVariableType end
"Links x to the weighted sum of breakpoints in SOS2 quadratic approximation."
struct SOS2LinkingConstraint <: ConstraintType end
struct SOS2LinkingExpression <: ExpressionType end
"Ensures the sum of λ weights equals 1 in SOS2 quadratic approximation."
struct SOS2NormConstraint <: ConstraintType end
struct SOS2NormExpression <: ExpressionType end

struct SolverSOS2Constraint <: ConstraintType end

"Config for solver-native SOS2 quadratic approximation (MOI.SOS2 adjacency)."
struct SolverSOS2QuadConfig <: QuadraticApproxConfig
    add_mccormick::Bool
end
SolverSOS2QuadConfig() = SolverSOS2QuadConfig(false)

"""
    _add_quadratic_approx!(config::SolverSOS2QuadConfig, container, C, names, time_steps, x_var, x_min, x_max, depth, meta)

Approximate x² using a piecewise linear function with solver-native SOS2 constraints.

Creates lambda_var (λ) variables representing convex combination weights over breakpoints,
adds linking, normalization, and MOI.SOS2 constraints, and stores affine expressions
approximating x² in a `QuadraticExpression` expression container.

# Arguments
- `config::SolverSOS2QuadConfig`: configuration (includes `add_mccormick` flag)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `depth::Int`: number of PWL segments
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function _add_quadratic_approx!(
    config::SolverSOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(x_min, x_max, _square; num_segments = depth)
    n_points = depth + 1
    jump_model = get_jump_model(container)

    # Create all containers upfront
    lambda_var =
        add_variable_container!(container, QuadraticVariable(), C; meta)
    link_cons = add_constraints_container!(
        container,
        SOS2LinkingConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    link_expr = add_expression_container!(
        container,
        SOS2LinkingExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    norm_cons = add_constraints_container!(
        container,
        SOS2NormConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    norm_expr = add_expression_container!(
        container,
        SOS2NormExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    sos_cons = add_constraints_container!(
        container,
        SolverSOS2Constraint(),
        C,
        names,
        time_steps;
        meta,
    )
    result_expr = add_expression_container!(
        container,
        QuadraticExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        x = x_var[name, t]

        # Create lambda_var variables: λ_i ∈ [0, 1]
        lambda = Vector{JuMP.VariableRef}(undef, n_points)
        for i in 1:n_points
            lambda[i] =
                lambda_var[(name, i, t)] = JuMP.@variable(
                    jump_model,
                    base_name = "QuadraticVariable_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = 1.0,
                )
        end

        # x = Σ λ_i * x_i
        link = link_expr[name, t] = JuMP.AffExpr(0.0)
        for i in eachindex(x_bkpts)
            add_proportional_to_jump_expression!(link, lambda[i], x_bkpts[i])
        end
        link_cons[name, t] = JuMP.@constraint(jump_model, x == link)

        # Σ λ_i = 1
        norm = norm_expr[name, t] = JuMP.AffExpr(0.0)
        for l in lambda
            add_proportional_to_jump_expression!(norm, l, 1.0)
        end
        norm_cons[name, t] = JuMP.@constraint(jump_model, norm == 1.0)

        # λ ∈ SOS2 (solver-native)
        sos_cons[name, t] =
            JuMP.@constraint(jump_model, lambda in MOI.SOS2(collect(1:n_points)))

        # Build x̂² = Σ λ_i * x_i² as an affine expression
        x_hat_sq = JuMP.AffExpr(0.0)
        for i in 1:n_points
            add_proportional_to_jump_expression!(x_hat_sq, lambda[i], x_sq_bkpts[i])
        end
        result_expr[name, t] = x_hat_sq
    end

    if config.add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, result_expr,
            x_min, x_max,
            meta,
        )
    end

    return result_expr
end
