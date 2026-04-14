# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses solver-native MOI.SOS2 constraints for adjacency enforcement.

"lambda_var (λ) convex combination weight variables for SOS2 quadratic approximation."
struct QuadraticVariable <: SparseVariableType end
"Links x to the weighted sum of breakpoints in SOS2 quadratic approximation."
struct SOS2LinkingConstraint <: ConstraintType end
"Expression for the weighted sum of breakpoints Σ λ_i * x_i linking x to lambda variables."
struct SOS2LinkingExpression <: ExpressionType end
"Ensures the sum of λ weights equals 1 in SOS2 quadratic approximation."
struct SOS2NormConstraint <: ConstraintType end
"Expression for the normalization sum Σ λ_i in SOS2 quadratic approximation."
struct SOS2NormExpression <: ExpressionType end

"Solver-native MOI.SOS2 adjacency constraint on lambda variables."
struct SolverSOS2Constraint <: ConstraintType end

"""
Config for solver-native SOS2 quadratic approximation (MOI.SOS2 adjacency).

# Fields
- `depth::Int`: number of PWL segments (breakpoints = depth + 1)
- `pwmcc_segments::Int`: number of piecewise McCormick cut partitions; 0 to disable (default 4)
"""
struct SolverSOS2QuadConfig <: QuadraticApproxConfig
    depth::Int
    pwmcc_segments::Int
end
SolverSOS2QuadConfig(depth::Int) = SolverSOS2QuadConfig(depth, 4)

"""
    _add_quadratic_approx!(config::SolverSOS2QuadConfig, jump_model, x, bounds, meta)

Inner (container-unaware) SOS2 quadratic approximation for a single (name, t) pair.

Computes a piecewise linear approximation of x² over [bounds.min, bounds.max] using
solver-native MOI.SOS2 adjacency constraints. Creates lambda (λ) convex-combination
weight variables, linking, normalization, and SOS2 constraints, and builds an affine
expression approximating x². Optionally adds piecewise McCormick concave cuts.

# Arguments
- `config::SolverSOS2QuadConfig`: configuration with `depth` and `pwmcc_segments`
- `jump_model::JuMP.Model`: the JuMP model to add variables and constraints to
- `x::JuMP.AbstractJuMPScalar`: the variable to approximate x² for
- `bounds::MinMax`: `(min = x_min, max = x_max)` domain bounds
- `meta::String`: base name prefix for JuMP variables and constraints

# Returns
A `NamedTuple` with fields:
- `lambda_vars::Vector{JuMP.VariableRef}` – n_points lambda weight variables
- `link_expr::JuMP.AffExpr` – weighted sum of breakpoints Σ λ_i * x_i
- `link_con::JuMP.ConstraintRef` – linking constraint (x − x_min)/lx == Σ λ_i * x_i
- `norm_expr::JuMP.AffExpr` – normalization sum Σ λ_i
- `norm_con::JuMP.ConstraintRef` – normalization constraint Σ λ_i == 1
- `sos_con::JuMP.ConstraintRef` – MOI.SOS2 adjacency constraint on λ
- `result_expr::JuMP.AffExpr` – affine expression approximating x²
- `pwmcc_result::Union{Nothing, NamedTuple}` – PWMCC cut result or nothing
"""
function _add_quadratic_approx!(
    config::SolverSOS2QuadConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    meta::String,
)
    lx = bounds.max - bounds.min
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(
            0.0,
            1.0,
            _square;
            num_segments = config.depth,
        )
    n_points = config.depth + 1

    # Create lambda variables: λ_i ∈ [0, 1]
    lambda = Vector{JuMP.VariableRef}(undef, n_points)
    for i in 1:n_points
        lambda[i] = JuMP.@variable(
            jump_model,
            base_name = "QuadraticVariable_$(meta)_pwl_$(i)",
            lower_bound = 0.0,
            upper_bound = 1.0,
        )
    end

    # x = Σ λ_i * x_i
    link_expr = JuMP.AffExpr(0.0)
    for i in eachindex(x_bkpts)
        add_proportional_to_jump_expression!(link_expr, lambda[i], x_bkpts[i])
    end
    link_con =
        JuMP.@constraint(jump_model, (x - bounds.min) / lx == link_expr)

    # Σ λ_i = 1
    norm_expr = JuMP.AffExpr(0.0)
    for l in lambda
        add_proportional_to_jump_expression!(norm_expr, l, 1.0)
    end
    norm_con = JuMP.@constraint(jump_model, norm_expr == 1.0)

    # λ ∈ SOS2 (solver-native)
    sos_con =
        JuMP.@constraint(jump_model, lambda in MOI.SOS2(collect(1:n_points)))

    # Build x̂² = Σ λ_i * x_i² as an affine expression
    x_hat_sq = JuMP.AffExpr(0.0)
    for i in 1:n_points
        add_proportional_to_jump_expression!(x_hat_sq, lambda[i], x_sq_bkpts[i])
    end
    result_expr = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(result_expr, x_hat_sq, lx * lx)
    add_proportional_to_jump_expression!(result_expr, x, 2 * bounds.min)
    add_constant_to_jump_expression!(result_expr, -bounds.min * bounds.min)

    pwmcc_result = nothing
    if config.pwmcc_segments > 0
        pwmcc_result = _add_pwmcc_concave_cuts!(
            jump_model, x, result_expr, bounds,
            config.pwmcc_segments, meta * "_pwmcc",
        )
    end

    return (
        lambda_vars = lambda,
        link_expr = link_expr,
        link_con = link_con,
        norm_expr = norm_expr,
        norm_con = norm_con,
        sos_con = sos_con,
        result_expr = result_expr,
        pwmcc_result = pwmcc_result,
    )
end

"""
    _create_quad_containers!(config::SolverSOS2QuadConfig, container, C, names, time_steps, meta)

Create optimization containers for solver-native SOS2 quadratic approximation (excluding
the result expression). Returns a `NamedTuple` of container references.
"""
function _create_quad_containers!(
    config::SolverSOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
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
    link_expr_c = add_expression_container!(
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
    norm_expr_c = add_expression_container!(
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
    return (
        lambda_var = lambda_var,
        link_cons = link_cons,
        link_expr_c = link_expr_c,
        norm_cons = norm_cons,
        norm_expr_c = norm_expr_c,
        sos_cons = sos_cons,
    )
end

"""
    _store_quad_result!(::SolverSOS2QuadConfig, containers, name, i, t, r)

Store the per-(name, t) result from the inner solver SOS2 function into the containers
created by `_create_quad_containers!`. Does not store `result_expr` or `pwmcc_result`.
"""
function _store_quad_result!(
    ::SolverSOS2QuadConfig,
    containers::NamedTuple,
    name::String,
    ::Int,
    t::Int,
    r::NamedTuple,
)
    for (j, lam) in enumerate(r.lambda_vars)
        containers.lambda_var[(name, j, t)] = lam
    end
    containers.link_expr_c[name, t] = r.link_expr
    containers.link_cons[name, t] = r.link_con
    containers.norm_expr_c[name, t] = r.norm_expr
    containers.norm_cons[name, t] = r.norm_con
    containers.sos_cons[name, t] = r.sos_con
    return nothing
end

"""
    add_quadratic_approx!(config::SolverSOS2QuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using a piecewise linear function with solver-native SOS2 constraints.

Creates all optimization containers, then delegates per-(name, t) work to the inner
[`_add_quadratic_approx!`](@ref) and populates lambda variables, linking/normalization
expressions and constraints, SOS2 constraints, and the quadratic expression result.
When `config.pwmcc_segments > 0`, also creates PWMCC containers and stores per-(name, t)
PWMCC cut results via [`_create_pwmcc_containers!`](@ref) and [`_store_pwmcc_result!`](@ref).

# Arguments
- `config::SolverSOS2QuadConfig`: configuration with `depth` (number of PWL segments) and `pwmcc_segments` (PWMCC cut partitions; 0 to disable, default 4)
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name bounds [(min=x_min, max=x_max), ...]
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function add_quadratic_approx!(
    config::SolverSOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    quad_containers = _create_quad_containers!(config, container, C, names, time_steps, meta)
    pwmcc_containers = nothing
    if config.pwmcc_segments > 0
        pwmcc_containers = _create_pwmcc_containers!(
            container, C, names, time_steps, config.pwmcc_segments, meta * "_pwmcc",
        )
    end
    result_expr = add_expression_container!(
        container,
        QuadraticExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    for (i, name) in enumerate(names), t in time_steps
        r = _add_quadratic_approx!(config, jump_model, x_var[name, t], bounds[i], meta)
        result_expr[name, t] = r.result_expr
        _store_quad_result!(config, quad_containers, name, i, t, r)
        if !isnothing(pwmcc_containers)
            _store_pwmcc_result!(pwmcc_containers, name, t, config.pwmcc_segments, r.pwmcc_result)
        end
    end
    return result_expr
end
