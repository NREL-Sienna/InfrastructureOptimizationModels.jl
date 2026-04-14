# Epigraph (Q^{L1}) LP-only lower bound for x² using tangent-line cuts.
# Pure LP — zero binary variables. Creates a variable z ≥ x² (approximately)
# bounded from below by supporting hyperplanes of the parabola.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024), Q^{L1} relaxation.

"Expression container for epigraph quadratic approximation results."
struct EpigraphExpression <: ExpressionType end

"Variable representing a lower-bounded approximation of x² in epigraph relaxation."
struct EpigraphVariable <: VariableType end
"Tangent-line lower-bound constraints in epigraph relaxation."
struct EpigraphTangentConstraint <: ConstraintType end
"Tangent-line lower-bound expression fL used in the epigraph formulation."
struct EpigraphTangentExpression <: ExpressionType end

"""
Config for epigraph (Q^{L1}) LP-only lower-bound quadratic approximation.

# Fields
- `depth::Int`: number of tangent-line breakpoints (2^depth + 1 tangent lines); pure LP, zero binary variables
"""
struct EpigraphQuadConfig <: QuadraticApproxConfig
    depth::Int
end

"""
    _add_quadratic_approx!(config::EpigraphQuadConfig, jump_model, x, bounds, meta)

Inner epigraph Q^{L1} lower bound for x² on a single (name, t).

Creates variable z ≥ 0 bounded by tangent-line cuts, plus auxiliary g variables.
Pure LP — zero binary variables. Container-unaware.

# Arguments
- `config::EpigraphQuadConfig`: configuration with `depth` field
- `jump_model::JuMP.Model`: JuMP model
- `x::JuMP.AbstractJuMPScalar`: single variable
- `bounds::MinMax`: (min, max) bounds for this name
- `meta::String`: base name prefix

# Returns
NamedTuple with fields:
- `g_vars::Vector{JuMP.VariableRef}` — auxiliary variables g_0,...,g_L
- `z_var::JuMP.VariableRef` — epigraph variable
- `lp_cons::Vector{JuMP.ConstraintRef}` — LP relaxation constraints
- `link_con::JuMP.ConstraintRef` — linking constraint
- `tangent_cons::Vector{JuMP.ConstraintRef}` — tangent line constraints
- `fL_expr::JuMP.AffExpr` — tangent expression
- `result_expr::JuMP.AffExpr` — result (= z as AffExpr)
"""
function _add_quadratic_approx!(
    config::EpigraphQuadConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    meta::String,
)
    IS.@assert_op bounds.max > bounds.min
    IS.@assert_op config.depth >= 1

    x_min = bounds.min
    x_max = bounds.max
    delta = x_max - x_min
    z_ub = max(x_min^2, x_max^2)

    # Auxiliary variables g_0,...,g_L ∈ [0, 1]
    g_vars = Vector{JuMP.VariableRef}(undef, config.depth + 1)
    for j in 0:(config.depth)
        g_vars[j + 1] = JuMP.@variable(
            jump_model,
            base_name = "SawtoothAux_$(meta)_$(j)",
            lower_bound = 0.0,
            upper_bound = 1.0,
        )
    end
    g0 = g_vars[1]  # g_0

    # Linking constraint: g_0 = (x - x_min) / Δ
    link_con = JuMP.@constraint(jump_model, g0 == (x - x_min) / delta)

    # LP constraints for j = 1,...,L
    lp_cons = JuMP.ConstraintRef[]
    for j in 1:(config.depth)
        g_prev = g_vars[j]      # g_{j-1}
        g_curr = g_vars[j + 1]  # g_j
        push!(lp_cons, JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev))
        push!(lp_cons, JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev)))
    end

    # Epigraph variable z
    z = JuMP.@variable(
        jump_model,
        base_name = "EpigraphVar_$(meta)",
        lower_bound = 0.0,
        upper_bound = z_ub,
    )

    # Tangent-line lower bounds
    fL = JuMP.AffExpr(0.0)
    tangent_cons = JuMP.ConstraintRef[]
    for j in 1:(config.depth)
        add_proportional_to_jump_expression!(
            fL,
            g_vars[j + 1],  # g_j
            delta * delta * 2.0^(-2j),
        )
        push!(tangent_cons, JuMP.@constraint(
            jump_model,
            z >= x_min * (2 * delta * g0 + x_min) - fL + delta^2 * (g0 - 2.0^(-2j - 2))
        ))
    end
    push!(tangent_cons, JuMP.@constraint(jump_model, z >= 0))
    push!(tangent_cons, JuMP.@constraint(
        jump_model,
        z >= 2.0 * x_min - 1.0 + 2.0 * delta * g0
    ))

    result_expr = JuMP.AffExpr(0.0, z => 1.0)

    return (
        g_vars = g_vars,
        z_var = z,
        lp_cons = lp_cons,
        link_con = link_con,
        tangent_cons = tangent_cons,
        fL_expr = fL,
        result_expr = result_expr,
    )
end

"""
    _create_epi_containers!(config, container, C, names, time_steps, meta)

Create containers for epigraph quadratic approximation results.

Returns a NamedTuple of container references (excluding `result_expr`,
which is created separately by the outer function).
"""
function _create_epi_containers!(
    config::EpigraphQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    g_levels = 0:(config.depth)
    z_var = add_variable_container!(container, EpigraphVariable(), C, names, time_steps; meta)
    g_var = add_variable_container!(
        container, SawtoothAuxVariable(), C, names, g_levels, time_steps; meta,
    )
    lp_cons = add_constraints_container!(
        container, SawtoothLPConstraint(), C, names, 1:2, time_steps; meta,
    )
    link_cons = add_constraints_container!(
        container, SawtoothLinkingConstraint(), C, names, time_steps; meta,
    )
    fL_expr = add_expression_container!(
        container, EpigraphTangentExpression(), C, names, time_steps; meta,
    )
    tangent_cons = add_constraints_container!(
        container, EpigraphTangentConstraint(), C, names, 1:(config.depth + 2), time_steps;
        sparse = true, meta,
    )
    return (
        z_var = z_var,
        g_var = g_var,
        lp_cons = lp_cons,
        link_cons = link_cons,
        fL_expr = fL_expr,
        tangent_cons = tangent_cons,
    )
end

"""
    _store_epi_result!(config, containers, name, i, t, result)

Store one (name, t) epigraph result into the pre-allocated containers.
"""
function _store_epi_result!(
    config::EpigraphQuadConfig,
    containers::NamedTuple,
    name::String,
    ::Int,
    t::Int,
    result::NamedTuple,
)
    g_levels = 0:(config.depth)
    containers.z_var[name, t] = result.z_var
    for (j_idx, j) in enumerate(g_levels)
        containers.g_var[name, j, t] = result.g_vars[j_idx]
    end
    containers.link_cons[name, t] = result.link_con
    for j in 1:(config.depth)
        containers.lp_cons[name, 1, t] = result.lp_cons[2 * (j - 1) + 1]
        containers.lp_cons[name, 2, t] = result.lp_cons[2 * (j - 1) + 2]
    end
    containers.fL_expr[name, t] = result.fL_expr
    for j in 1:(config.depth)
        containers.tangent_cons[(name, j + 1, t)] = result.tangent_cons[j]
    end
    containers.tangent_cons[name, 1, t] = result.tangent_cons[config.depth + 1]
    containers.tangent_cons[name, config.depth + 1, t] = result.tangent_cons[config.depth + 2]
    return nothing
end

"""
    add_quadratic_approx!(config::EpigraphQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Epigraph Q^{L1} LP-only lower bound for x² using tangent-line cuts.

Creates containers for all epigraph components and populates them via single (name, t) loop.

# Arguments
- `config::EpigraphQuadConfig`: configuration with `depth` field
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name bounds [(min=x_min, max=x_max), ...]
- `meta::String`: variable type identifier
"""
function add_quadratic_approx!(
    config::EpigraphQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    epi_containers = _create_epi_containers!(config, container, C, names, time_steps, meta)
    result_expr = add_expression_container!(
        container, EpigraphExpression(), C, names, time_steps; meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        r = _add_quadratic_approx!(config, jump_model, x_var[name, t], bounds[i], meta)
        result_expr[name, t] = r.result_expr
        _store_epi_result!(config, epi_containers, name, i, t, r)
    end

    return result_expr
end
