# Sawtooth MIP approximation of x² for use in constraints.
# Uses recursive tooth function compositions with O(log(1/ε)) binary variables.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024).

"Auxiliary continuous variables (g₀, …, g_L) for sawtooth quadratic approximation."
struct SawtoothAuxVariable <: VariableType end
"Binary variables (α₁, …, α_L) for sawtooth quadratic approximation."
struct SawtoothBinaryVariable <: VariableType end
"Variable result in tightened version."
struct SawtoothTightenedVariable <: VariableType end
"Links g₀ to the normalized x value in sawtooth quadratic approximation."
struct SawtoothLinkingConstraint <: ConstraintType end
"Constrains g_j based on g_{j-1}."
struct SawtoothMIPConstraint <: ConstraintType end
"LP relaxation constraints (g_j ≤ 2g_{j-1}, g_j ≤ 2(1−g_{j-1})) used in epigraph tightening."
struct SawtoothLPConstraint <: ConstraintType end
"Bounds tightened variable."
struct SawtoothTightenedConstraint <: ConstraintType end

"""
Config for sawtooth MIP quadratic approximation.

# Fields
- `depth::Int`: recursion depth L; uses L binary variables for 2^L + 1 breakpoints
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound; 0 to disable (default 0)
"""
struct SawtoothQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
SawtoothQuadConfig(depth::Int) = SawtoothQuadConfig(depth, 0)

"""
    _add_quadratic_approx!(config::SawtoothQuadConfig, model, x, bounds, meta)

Inner (container-unaware) sawtooth MIP approximation of x².

Creates auxiliary continuous variables g_0,...,g_L and binary variables α_1,...,α_L,
adds S^L constraints (4 per level) and a linking constraint, and returns an affine
expression approximating x².

If `config.epigraph_depth > 0`, also creates a tightened variable z bounded between
the sawtooth upper bound and an epigraph LP lower bound.

# Arguments
- `config::SawtoothQuadConfig`: configuration with `depth` and `epigraph_depth`
- `jump_model::JuMP.Model`: the JuMP model
- `x::JuMP.AbstractJuMPScalar`: the variable to square
- `bounds::MinMax`: `(min = x_min, max = x_max)` domain bounds
- `meta::String`: base name prefix for variables and constraints
"""
function _add_quadratic_approx!(
    config::SawtoothQuadConfig,
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
    alpha_levels = 1:(config.depth)

    # Auxiliary variables g_0,...,g_L ∈ [0, 1]
    g_vars = Vector{JuMP.VariableRef}(undef, config.depth + 1)
    for j in 0:(config.depth)
        g_vars[j + 1] = JuMP.@variable(
            jump_model,
            base_name = "$(meta)_SawtoothAux_{$(j)}",
            lower_bound = 0.0,
            upper_bound = 1.0,
        )
    end

    # Binary variables α_1,...,α_L
    alpha_vars = Vector{JuMP.VariableRef}(undef, config.depth)
    for j in alpha_levels
        alpha_vars[j] = JuMP.@variable(
            jump_model,
            base_name = "$(meta)_SawtoothBin_{$(j)}",
            binary = true,
        )
    end

    # Linking constraint: g_0 = (x - x_min) / Δ
    link_con = JuMP.@constraint(
        jump_model,
        g_vars[1] == (x - x_min) / delta,
    )

    # S^L constraints for j = 1,...,L (4 per level, flattened)
    mip_cons = Vector{JuMP.ConstraintRef}(undef, 4 * config.depth)
    for j in alpha_levels
        g_prev = g_vars[j]       # g_{j-1} is at index j (1-indexed)
        g_curr = g_vars[j + 1]   # g_j is at index j+1
        alpha_j = alpha_vars[j]
        offset = 4 * (j - 1)

        # g_j ≤ 2 g_{j-1}
        mip_cons[offset + 1] = JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev)
        # g_j ≤ 2(1 - g_{j-1})
        mip_cons[offset + 2] =
            JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev))
        # g_j ≥ 2(g_{j-1} - α_j)
        mip_cons[offset + 3] =
            JuMP.@constraint(jump_model, g_curr >= 2.0 * (g_prev - alpha_j))
        # g_j ≥ 2(α_j - g_{j-1})
        mip_cons[offset + 4] =
            JuMP.@constraint(jump_model, g_curr >= 2.0 * (alpha_j - g_prev))
    end

    # Build x² ≈ x_min² + (2 x_min Δ + Δ²) g_0 - Σ_{j=1}^L Δ² 2^{-2j} g_j
    x_sq_approx = JuMP.AffExpr(x_min * x_min)
    add_proportional_to_jump_expression!(
        x_sq_approx,
        g_vars[1],
        2.0 * x_min * delta + delta * delta,
    )
    for j in alpha_levels
        add_proportional_to_jump_expression!(
            x_sq_approx,
            g_vars[j + 1],
            -delta * delta * (2.0^(-2 * j)),
        )
    end

    # Epigraph tightening
    z_var = nothing
    tight_cons = nothing
    epi_result = nothing
    result_expr = x_sq_approx

    if config.epigraph_depth > 0
        epi_result = _add_quadratic_approx!(
            EpigraphQuadConfig(config.epigraph_depth),
            jump_model, x, bounds, meta * "_lb",
        )

        z_min = (x_min <= 0.0 <= x_max) ? 0.0 : min(x_min^2, x_max^2)
        z_max = max(x_min^2, x_max^2)
        z_var = JuMP.@variable(
            jump_model,
            base_name = "$(meta)_TightenedSawtooth",
            lower_bound = z_min,
            upper_bound = z_max,
        )
        tight_cons = Vector{JuMP.ConstraintRef}(undef, 2)
        tight_cons[1] = JuMP.@constraint(jump_model, z_var <= x_sq_approx)
        tight_cons[2] = JuMP.@constraint(jump_model, z_var >= epi_result.result_expr)
        result_expr = JuMP.AffExpr(0.0, z_var => 1.0)
    end

    return (
        g_vars = g_vars,
        alpha_vars = alpha_vars,
        mip_cons = mip_cons,
        link_con = link_con,
        result_expr = result_expr,
        z_var = z_var,
        tight_cons = tight_cons,
        epi_result = epi_result,
    )
end

"""
    _create_quad_containers!(config::SawtoothQuadConfig, container, C, names, time_steps, meta)

Create optimization containers for sawtooth quadratic approximation (excluding the
result expression). Returns a `NamedTuple` of container references.
"""
function _create_quad_containers!(
    config::SawtoothQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    depth = config.depth
    g_var = add_variable_container!(
        container,
        SawtoothAuxVariable(),
        C,
        names,
        0:depth,
        time_steps;
        meta,
    )
    alpha_var = add_variable_container!(
        container,
        SawtoothBinaryVariable(),
        C,
        names,
        1:depth,
        time_steps;
        meta,
    )
    mip_cons = add_constraints_container!(
        container,
        SawtoothMIPConstraint(),
        C,
        names,
        1:4,
        time_steps;
        sparse = true,
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        SawtoothLinkingConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    z_var = nothing
    tight_cons = nothing
    if config.epigraph_depth > 0
        z_var = add_variable_container!(
            container,
            SawtoothTightenedVariable(),
            C,
            names,
            time_steps;
            meta,
        )
        tight_cons = add_constraints_container!(
            container,
            SawtoothTightenedConstraint(),
            C,
            names,
            1:2,
            time_steps;
            meta,
        )
    end
    return (
        g_var = g_var,
        alpha_var = alpha_var,
        mip_cons = mip_cons,
        link_cons = link_cons,
        z_var = z_var,
        tight_cons = tight_cons,
    )
end

"""
    _store_quad_result!(config::SawtoothQuadConfig, containers, name, i, t, r)

Store the per-(name, t) result from the inner sawtooth function into the containers
created by `_create_quad_containers!`. Does not store `result_expr` or `epi_result`.
"""
function _store_quad_result!(
    config::SawtoothQuadConfig,
    containers::NamedTuple,
    name::String,
    ::Int,
    t::Int,
    r::NamedTuple,
)
    depth = config.depth
    for j in 0:depth
        containers.g_var[name, j, t] = r.g_vars[j + 1]
    end
    for j in 1:depth
        containers.alpha_var[name, j, t] = r.alpha_vars[j]
    end
    for j in 1:depth
        offset = 4 * (j - 1)
        containers.mip_cons[name, 1, t] = r.mip_cons[offset + 1]
        containers.mip_cons[name, 2, t] = r.mip_cons[offset + 2]
        containers.mip_cons[name, 3, t] = r.mip_cons[offset + 3]
        containers.mip_cons[name, 4, t] = r.mip_cons[offset + 4]
    end
    containers.link_cons[name, t] = r.link_con
    if !isnothing(r.z_var)
        containers.z_var[name, t] = r.z_var
        containers.tight_cons[name, 1, t] = r.tight_cons[1]
        containers.tight_cons[name, 2, t] = r.tight_cons[2]
    end
    return nothing
end

"""
    add_quadratic_approx!(config::SawtoothQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Outer (container-aware) sawtooth MIP approximation of x².

Creates all optimization containers upfront, then calls the inner
`_add_quadratic_approx!` per (name, time step) to populate them.
When `config.epigraph_depth > 0`, also creates and populates epigraph
containers for the LP lower-bound tightening.

# Arguments
- `config::SawtoothQuadConfig`: configuration with `depth` and `epigraph_depth`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name bounds [(min=x_min, max=x_max), ...]
- `meta::String`: variable type identifier for the approximation
"""
function add_quadratic_approx!(
    config::SawtoothQuadConfig,
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
    epi_containers = nothing
    epi_cfg = nothing
    if config.epigraph_depth > 0
        epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
        epi_containers = _create_epi_containers!(epi_cfg, container, C, names, time_steps, meta * "_lb")
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
        if !isnothing(epi_containers)
            _store_epi_result!(epi_cfg, epi_containers, name, i, t, r.epi_result)
        end
    end
    return result_expr
end
