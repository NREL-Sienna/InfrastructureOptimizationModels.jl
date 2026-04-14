# SOS2-based piecewise linear approximation of x² for use in constraints.
# Uses manually-implemented SOS2 adjacency via binary variables and linear constraints.

"Binary segment-selection variables (z) for manual SOS2 quadratic approximation."
struct ManualSOS2BinaryVariable <: SparseVariableType end
"Ensures exactly one segment is active (∑zⱼ = 1) in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionConstraint <: ConstraintType end
"Expression for the segment selection sum Σ z_j in manual SOS2 quadratic approximation."
struct ManualSOS2SegmentSelectionExpression <: ExpressionType end
"Links active segment to lambda variables."
struct ManualSOS2AdjacencyConstraint <: ConstraintType end

"""
Config for manual binary-variable SOS2 quadratic approximation.

# Fields
- `depth::Int`: number of PWL segments (breakpoints = depth + 1)
- `pwmcc_segments::Int`: number of piecewise McCormick cut partitions; 0 to disable (default 4)
"""
struct ManualSOS2QuadConfig <: QuadraticApproxConfig
    depth::Int
    pwmcc_segments::Int
end
ManualSOS2QuadConfig(depth::Int) = ManualSOS2QuadConfig(depth, 4)

"""
    _add_quadratic_approx!(config::ManualSOS2QuadConfig, jump_model, x, bounds, meta)

Inner function: approximate x² for a single variable using manual SOS2 constraints.

Creates lambda (λ) weight variables, binary segment-selection variables (z),
linking, normalization, segment-selection, and adjacency constraints,
and builds an affine expression approximating x².

Optionally adds piecewise McCormick concave cuts if `config.pwmcc_segments > 0`.

# Arguments
- `config::ManualSOS2QuadConfig`: configuration with `depth` and `pwmcc_segments`
- `jump_model::JuMP.Model`: the JuMP model
- `x::JuMP.AbstractJuMPScalar`: the variable to approximate x² for
- `bounds::MinMax`: `(min = x_min, max = x_max)` domain bounds
- `meta::String`: base name prefix for JuMP variables and constraints

# Returns
A `NamedTuple` with fields:
- `lambda_vars::Vector{JuMP.VariableRef}` – length `n_points` (depth + 1)
- `z_vars::Vector{JuMP.VariableRef}` – length `n_bins` (depth)
- `link_expr::JuMP.AffExpr` – linking expression Σ λ_i * x_i
- `link_con::JuMP.ConstraintRef` – linking constraint (x − x_min)/lx == Σ λ_i * x_i
- `norm_expr::JuMP.AffExpr` – normalization expression Σ λ_i
- `norm_con::JuMP.ConstraintRef` – normalization constraint Σ λ_i == 1
- `seg_expr::JuMP.AffExpr` – segment selection expression Σ z_j
- `seg_con::JuMP.ConstraintRef` – segment selection constraint Σ z_j == 1
- `adj_cons::Vector{JuMP.ConstraintRef}` – length `n_points` adjacency constraints
- `result_expr::JuMP.AffExpr` – affine expression approximating x²
- `pwmcc_result::Union{Nothing, NamedTuple}` – result from `_add_pwmcc_concave_cuts!`, or `nothing`
"""
function _add_quadratic_approx!(
    config::ManualSOS2QuadConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    meta::String,
)
    x_min = bounds.min
    x_max = bounds.max
    lx = x_max - x_min
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(
            0.0,
            1.0,
            _square;
            num_segments = config.depth,
        )
    n_points = config.depth + 1
    n_bins = n_points - 1

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

    # (x − x_min) / lx == Σ λ_i * x_i
    link_expr = JuMP.AffExpr(0.0)
    for i in eachindex(x_bkpts)
        add_proportional_to_jump_expression!(link_expr, lambda[i], x_bkpts[i])
    end
    link_con = JuMP.@constraint(jump_model, (x - x_min) / lx == link_expr)

    # Σ λ_i = 1
    norm_expr = JuMP.AffExpr(0.0)
    for l in lambda
        add_proportional_to_jump_expression!(norm_expr, l, 1.0)
    end
    norm_con = JuMP.@constraint(jump_model, norm_expr == 1.0)

    # Create binary segment-selection variables z_j
    z_vars = Vector{JuMP.VariableRef}(undef, n_bins)
    for j in 1:n_bins
        z_vars[j] = JuMP.@variable(
            jump_model,
            base_name = "ManualSOS2Binary_$(meta)_$(j)",
            binary = true,
        )
    end

    # Σ z_j = 1 (segment selection)
    seg_expr = JuMP.AffExpr(0.0)
    for z in z_vars
        add_proportional_to_jump_expression!(seg_expr, z, 1.0)
    end
    seg_con = JuMP.@constraint(jump_model, seg_expr == 1)

    # Adjacency constraints: λ_i ≤ z_{i-1} + z_i (with boundary cases)
    adj_cons = Vector{JuMP.ConstraintRef}(undef, n_points)
    adj_cons[1] = JuMP.@constraint(jump_model, lambda[1] <= z_vars[1])
    for i in 2:(n_points - 1)
        adj_cons[i] =
            JuMP.@constraint(jump_model, lambda[i] <= z_vars[i - 1] + z_vars[i])
    end
    adj_cons[n_points] =
        JuMP.@constraint(jump_model, lambda[n_points] <= z_vars[n_bins])

    # Build x̂² = Σ λ_i * x_i² as an affine expression
    x_hat_sq = JuMP.AffExpr(0.0)
    for i in 1:n_points
        add_proportional_to_jump_expression!(x_hat_sq, lambda[i], x_sq_bkpts[i])
    end
    x_sq = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(x_sq, x_hat_sq, lx * lx)
    add_proportional_to_jump_expression!(x_sq, x, 2 * x_min)
    add_constant_to_jump_expression!(x_sq, -x_min * x_min)

    pwmcc_result = nothing
    if config.pwmcc_segments > 0
        pwmcc_result = _add_pwmcc_concave_cuts!(
            jump_model, x, x_sq, bounds,
            config.pwmcc_segments, meta * "_pwmcc",
        )
    end

    return (
        lambda_vars = lambda,
        z_vars = z_vars,
        link_expr = link_expr,
        link_con = link_con,
        norm_expr = norm_expr,
        norm_con = norm_con,
        seg_expr = seg_expr,
        seg_con = seg_con,
        adj_cons = adj_cons,
        result_expr = x_sq,
        pwmcc_result = pwmcc_result,
    )
end

"""
    _create_quad_containers!(config::ManualSOS2QuadConfig, container, C, names, time_steps, meta)

Create optimization containers for manual SOS2 quadratic approximation (excluding the
result expression). Returns a `NamedTuple` of container references.
"""
function _create_quad_containers!(
    config::ManualSOS2QuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    n_points = config.depth + 1
    lambda_container =
        add_variable_container!(container, QuadraticVariable(), C; meta)
    z_container = add_variable_container!(container, ManualSOS2BinaryVariable(), C; meta)
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
    seg_cons = add_constraints_container!(
        container,
        ManualSOS2SegmentSelectionConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    seg_expr_c = add_expression_container!(
        container,
        ManualSOS2SegmentSelectionExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    adj_cons = add_constraints_container!(
        container,
        ManualSOS2AdjacencyConstraint(),
        C,
        names,
        1:n_points,
        time_steps;
        meta,
    )
    return (
        lambda_container = lambda_container,
        z_container = z_container,
        link_cons = link_cons,
        link_expr_c = link_expr_c,
        norm_cons = norm_cons,
        norm_expr_c = norm_expr_c,
        seg_cons = seg_cons,
        seg_expr_c = seg_expr_c,
        adj_cons = adj_cons,
    )
end

"""
    _store_quad_result!(::ManualSOS2QuadConfig, containers, name, i, t, r)

Store the per-(name, t) result from the inner manual SOS2 function into the containers
created by `_create_quad_containers!`. Does not store `result_expr` or `pwmcc_result`.
"""
function _store_quad_result!(
    ::ManualSOS2QuadConfig,
    containers::NamedTuple,
    name::String,
    ::Int,
    t::Int,
    r::NamedTuple,
)
    for (j, lam) in enumerate(r.lambda_vars)
        containers.lambda_container[(name, j, t)] = lam
    end
    for (j, z) in enumerate(r.z_vars)
        containers.z_container[(name, j, t)] = z
    end
    containers.link_expr_c[name, t] = r.link_expr
    containers.link_cons[name, t] = r.link_con
    containers.norm_expr_c[name, t] = r.norm_expr
    containers.norm_cons[name, t] = r.norm_con
    containers.seg_expr_c[name, t] = r.seg_expr
    containers.seg_cons[name, t] = r.seg_con
    for (j, ac) in enumerate(r.adj_cons)
        containers.adj_cons[name, j, t] = ac
    end
    return nothing
end

"""
    add_quadratic_approx!(config::ManualSOS2QuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Approximate x² using manual SOS2 for all components and time steps.

Creates optimization containers for all variable/constraint/expression types,
loops over (name, t) pairs calling the inner `_add_quadratic_approx!`,
and populates all containers from the results. When `config.pwmcc_segments > 0`,
also creates and populates PWMCC containers for the concave-cut results.

# Arguments
- `config::ManualSOS2QuadConfig`: configuration with `depth` and `pwmcc_segments`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name bounds `[(min=x_min, max=x_max), ...]`
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function add_quadratic_approx!(
    config::ManualSOS2QuadConfig,
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
