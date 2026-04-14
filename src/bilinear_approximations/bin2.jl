# Bin2 separable approximation of bilinear products z = x·y.
# Uses the identity: x·y = (1/2)*((x+y)² − x² - y²).
# Calls existing quadratic approximation functions for p²=(x+y)²

"Expression container for bilinear product (x·y) approximation results."
struct BilinearProductExpression <: ExpressionType end
"Variable container for bilinear product (x ̇y) approximation results."
struct BilinearProductVariable <: VariableType end
"Expression container for adding variables."
struct VariableSumExpression <: ExpressionType end
"Expression container for subtracting variables."
struct VariableDifferenceExpression <: ExpressionType end
"Constraint container for linking product expressions and variables."
struct BilinearProductLinkingConstraint <: ConstraintType end

# --- Bilinear approximation config hierarchy ---

"Abstract supertype for bilinear approximation method configurations."
abstract type BilinearApproxConfig end

"""
Config for Bin2 bilinear approximation using z = ½((x+y)² − x² − y²).

# Fields
- `quad_config::QuadraticApproxConfig`: quadratic method used for x², y², and (x+y)²
- `add_mccormick::Bool`: whether to add reformulated McCormick cuts through separable variables (default true)
"""
struct Bin2Config <: BilinearApproxConfig
    quad_config::QuadraticApproxConfig
    add_mccormick::Bool
end
Bin2Config(quad_config::QuadraticApproxConfig) = Bin2Config(quad_config, true)

# --- Inner (JuMP-level) Bin2 bilinear approximation ---

"""
    _add_bilinear_approx!(config::Bin2Config, jump_model, x, y, xsq, ysq, x_bounds, y_bounds, meta)

Inner (precomputed) Bin2 bilinear approximation for a single (name, t) pair.

Uses the identity z = ½((x+y)² − x² − y²) with pre-computed quadratic approximations
`xsq` ≈ x² and `ysq` ≈ y². Optionally adds 4 reformulated McCormick constraints.

# Arguments
- `config::Bin2Config`: configuration with `quad_config` and `add_mccormick`
- `jump_model::JuMP.Model`: the JuMP model
- `x::JuMP.AbstractJuMPScalar`: first variable in the bilinear product
- `y::JuMP.AbstractJuMPScalar`: second variable in the bilinear product
- `xsq::JuMP.AbstractJuMPScalar`: pre-computed x² approximation
- `ysq::JuMP.AbstractJuMPScalar`: pre-computed y² approximation
- `x_bounds::MinMax`: `(min, max)` domain bounds for x
- `y_bounds::MinMax`: `(min, max)` domain bounds for y
- `meta::String`: base name prefix for variables and constraints
"""
function _add_bilinear_approx!(
    config::Bin2Config,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    xsq::JuMP.AbstractJuMPScalar,
    ysq::JuMP.AbstractJuMPScalar,
    x_bounds::MinMax,
    y_bounds::MinMax,
    meta::String,
)
    # p = x + y
    p = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(p, x, 1.0)
    add_proportional_to_jump_expression!(p, y, 1.0)

    p_bounds = (min = x_bounds.min + y_bounds.min, max = x_bounds.max + y_bounds.max)

    # Approximate p² = (x+y)² using the inner quadratic
    psq_r = _add_quadratic_approx!(
        config.quad_config, jump_model, p, p_bounds, meta * "_plus",
    )
    psq = psq_r.result_expr

    # z = ½(p² − x² − y²)
    result = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(result, psq, 0.5)
    add_proportional_to_jump_expression!(result, xsq, -0.5)
    add_proportional_to_jump_expression!(result, ysq, -0.5)

    # Reformulated McCormick cuts (4 inline constraints)
    mc_cons = nothing
    if config.add_mccormick
        x_min = x_bounds.min
        x_max = x_bounds.max
        y_min = y_bounds.min
        y_max = y_bounds.max
        mc_cons = Vector{JuMP.ConstraintRef}(undef, 4)
        mc_cons[1] = JuMP.@constraint(
            jump_model,
            psq - xsq - ysq >= 2.0 * (x_min * y + x * y_min - x_min * y_min),
        )
        mc_cons[2] = JuMP.@constraint(
            jump_model,
            psq - xsq - ysq >= 2.0 * (x_max * y + x * y_max - x_max * y_max),
        )
        mc_cons[3] = JuMP.@constraint(
            jump_model,
            psq - xsq - ysq <= 2.0 * (x_max * y + x * y_min - x_max * y_min),
        )
        mc_cons[4] = JuMP.@constraint(
            jump_model,
            psq - xsq - ysq <= 2.0 * (x_min * y + x * y_max - x_min * y_max),
        )
    end

    return (
        p_expr = p,
        psq_result = psq_r,
        result_expr = result,
        mc_cons = mc_cons,
    )
end

# --- Outer (container-level) Bin2 bilinear approximation ---

"""
    add_bilinear_approx!(config::Bin2Config, container, C, names, time_steps, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Outer (precomputed) Bin2 bilinear approximation across all (name, t) pairs.

Accepts pre-computed quadratic containers `xsq` ≈ x² and `ysq` ≈ y², creates
result containers, and delegates each (name, t) to the inner `_add_bilinear_approx!`.
All inner results are stored in containers:

- `BilinearProductExpression`: z = ½((x+y)² − x² − y²) result
- `ReformulatedMcCormickConstraint`: 4 McCormick cuts (when `add_mccormick`)
- `VariableSumExpression`: p = x + y expression
- `QuadraticExpression` (meta `"_plus"`): p² result expression
- Quad-config–specific containers for p² internals (via `_create_quad_containers!`)
- Nested epigraph containers when `quad_config` is `SawtoothQuadConfig` with `epigraph_depth > 0`
- Nested PWMCC containers when `quad_config` is `ManualSOS2QuadConfig`/`SolverSOS2QuadConfig` with `pwmcc_segments > 0`
- Standalone epigraph containers when `quad_config` is `EpigraphQuadConfig`

# Arguments
- `config::Bin2Config`: configuration with `quad_config` and `add_mccormick`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `xsq`: pre-computed x² approximation container indexed by `[name, t]`
- `ysq`: pre-computed y² approximation container indexed by `[name, t]`
- `x_var`: container of x variables indexed by `[name, t]`
- `y_var`: container of y variables indexed by `[name, t]`
- `x_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for x
- `y_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_bilinear_approx!(
    config::Bin2Config,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    xsq,
    ysq,
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    result_expr = add_expression_container!(
        container, BilinearProductExpression(), C, names, time_steps; meta,
    )
    if config.add_mccormick
        mc_cons = add_constraints_container!(
            container, ReformulatedMcCormickConstraint(), C, names, 1:4, time_steps;
            sparse = true, meta,
        )
    end

    # p = x + y expression container
    p_expr_c = add_expression_container!(
        container, VariableSumExpression(), C, names, time_steps; meta,
    )

    # Quad containers for p² (dispatches on config.quad_config type)
    psq_quad_containers = nothing
    if !(config.quad_config isa EpigraphQuadConfig)
        psq_quad_containers = _create_quad_containers!(
            config.quad_config, container, C, names, time_steps, meta * "_plus",
        )
    end
    psq_result_expr = add_expression_container!(
        container, QuadraticExpression(), C, names, time_steps; meta = meta * "_plus",
    )

    # For SawtoothQuadConfig with epigraph tightening
    psq_epi_containers = nothing
    psq_epi_cfg = nothing
    if config.quad_config isa SawtoothQuadConfig && config.quad_config.epigraph_depth > 0
        psq_epi_cfg = EpigraphQuadConfig(config.quad_config.epigraph_depth)
        psq_epi_containers = _create_epi_containers!(
            psq_epi_cfg, container, C, names, time_steps, meta * "_plus_lb",
        )
    end

    # For ManualSOS2/SolverSOS2 with PWMCC cuts
    psq_pwmcc_containers = nothing
    qc = config.quad_config
    if (qc isa ManualSOS2QuadConfig || qc isa SolverSOS2QuadConfig) &&
       qc.pwmcc_segments > 0
        psq_pwmcc_containers = _create_pwmcc_containers!(
            container, C, names, time_steps, qc.pwmcc_segments, meta * "_plus_pwmcc",
        )
    end

    # For EpigraphQuadConfig as quad_config (standalone epigraph)
    psq_standalone_epi_containers = nothing
    psq_standalone_epi_cfg = nothing
    if config.quad_config isa EpigraphQuadConfig
        psq_standalone_epi_cfg = config.quad_config
        psq_standalone_epi_containers = _create_epi_containers!(
            psq_standalone_epi_cfg, container, C, names, time_steps, meta * "_plus",
        )
    end

    for (i, name) in enumerate(names), t in time_steps
        r = _add_bilinear_approx!(
            config, jump_model, x_var[name, t], y_var[name, t],
            xsq[name, t], ysq[name, t],
            x_bounds[i], y_bounds[i], meta,
        )
        result_expr[name, t] = r.result_expr
        if config.add_mccormick && r.mc_cons !== nothing
            for k in 1:4
                mc_cons[name, k, t] = r.mc_cons[k]
            end
        end

        p_expr_c[name, t] = r.p_expr
        psq_result_expr[name, t] = r.psq_result.result_expr

        # Store quad-specific fields (varies by config type, dispatch handles it)
        if !isnothing(psq_quad_containers)
            _store_quad_result!(
                config.quad_config, psq_quad_containers, name, i, t, r.psq_result,
            )
        end

        # Store nested epi result (sawtooth tightening)
        if !isnothing(psq_epi_containers)
            _store_epi_result!(
                psq_epi_cfg, psq_epi_containers, name, i, t, r.psq_result.epi_result,
            )
        end

        # Store nested pwmcc result
        if !isnothing(psq_pwmcc_containers)
            _store_pwmcc_result!(
                psq_pwmcc_containers, name, t, qc.pwmcc_segments,
                r.psq_result.pwmcc_result,
            )
        end

        # Store standalone epi result (when quad_config is EpigraphQuadConfig)
        if !isnothing(psq_standalone_epi_containers)
            _store_epi_result!(
                psq_standalone_epi_cfg, psq_standalone_epi_containers, name, i, t,
                r.psq_result,
            )
        end
    end
    return result_expr
end

"""
    add_bilinear_approx!(config::Bin2Config, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Outer (non-precomputed) Bin2 bilinear approximation across all (name, t) pairs.

Computes x² and y² via the outer `add_quadratic_approx!` (which creates its own
containers), then delegates to the precomputed outer form.

# Arguments
- `config::Bin2Config`: configuration with `quad_config` and `add_mccormick`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by `[name, t]`
- `y_var`: container of y variables indexed by `[name, t]`
- `x_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for x
- `y_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_bilinear_approx!(
    config::Bin2Config,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    xsq = add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        x_var, x_bounds, meta * "_x",
    )
    ysq = add_quadratic_approx!(
        config.quad_config, container, C, names, time_steps,
        y_var, y_bounds, meta * "_y",
    )
    return add_bilinear_approx!(
        config, container, C, names, time_steps,
        xsq, ysq, x_var, y_var,
        x_bounds, y_bounds, meta,
    )
end
