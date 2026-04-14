# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines Bin2 lower bound and Bin3 upper bound with shared sawtooth for x², y²
# and LP-only epigraph for (x+y)², (x−y)². Uses 2L binaries instead of 3L (Bin2).
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
Config for HybS (Hybrid Separable) bilinear approximation.

Combines Bin2 lower bound and Bin3 upper bound with shared quadratic for x², y²
and LP-only epigraph for (x+y)², (x−y)².

# Fields
- `quad_config::QuadraticApproxConfig`: quadratic method used for the shared x² and y² terms
- `epigraph_depth::Int`: depth for the epigraph Q^{L1} LP-only approximation of cross-terms (x±y)²
- `add_mccormick::Bool`: whether to add standard McCormick envelope cuts on the product variable (default false)
"""
struct HybSConfig <: BilinearApproxConfig
    quad_config::QuadraticApproxConfig
    epigraph_depth::Int
    add_mccormick::Bool
end
HybSConfig(quad_config::QuadraticApproxConfig, epigraph_depth::Int) =
    HybSConfig(quad_config, epigraph_depth, false)

# --- Inner (container-unaware) HybS bilinear approximation ---

"""
    _add_bilinear_approx!(config::HybSConfig, model, x, y, xsq, ysq, x_bounds, y_bounds, meta)

Inner (container-unaware) HybS bilinear approximation for a single (name, t).

Combines Bin2 and Bin3 separable identities with pre-computed x² and y² approximations:
- Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y) where z_p1 lower-bounds (x+y)²
- Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2) where z_p2 lower-bounds (x−y)²

The cross-terms (x+y)² and (x−y)² use epigraph Q^{L1} (pure LP).

# Arguments
- `config::HybSConfig`: HybS configuration
- `jump_model::JuMP.Model`: the JuMP model
- `x::JuMP.AbstractJuMPScalar`: the x variable
- `y::JuMP.AbstractJuMPScalar`: the y variable
- `xsq::JuMP.AbstractJuMPScalar`: pre-computed approximation of x²
- `ysq::JuMP.AbstractJuMPScalar`: pre-computed approximation of y²
- `x_bounds::MinMax`: `(min, max)` bounds for x
- `y_bounds::MinMax`: `(min, max)` bounds for y
- `meta::String`: base name prefix for variables and constraints

# Returns
Named tuple with fields:
- `p1_expr`: affine expression for x + y
- `p2_expr`: affine expression for x − y
- `zp1_result`: result from inner epigraph for (x+y)²
- `zp2_result`: result from inner epigraph for (x−y)²
- `z_var`: product variable z ≈ x·y
- `hybrid_cons`: tuple of (lower_con, upper_con) for Bin2/Bin3 bounds
- `result_expr`: affine expression equal to z
- `mc_cons`: McCormick constraints (4 elements) or `nothing`
"""
function _add_bilinear_approx!(
    config::HybSConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    xsq::JuMP.AbstractJuMPScalar,
    ysq::JuMP.AbstractJuMPScalar,
    x_bounds::MinMax,
    y_bounds::MinMax,
    meta::String,
)
    # p1 = x + y, p2 = x − y
    p1 = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(p1, x, 1.0)
    add_proportional_to_jump_expression!(p1, y, 1.0)

    p2 = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(p2, x, 1.0)
    add_proportional_to_jump_expression!(p2, y, -1.0)

    p1_bounds = (min = x_bounds.min + y_bounds.min, max = x_bounds.max + y_bounds.max)
    p2_bounds = (min = x_bounds.min - y_bounds.max, max = x_bounds.max - y_bounds.min)

    # Epigraph Q^{L1} lower bounds for (x+y)² and (x−y)²
    epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
    zp1_r = _add_quadratic_approx!(epi_cfg, jump_model, p1, p1_bounds, meta * "_plus")
    zp2_r = _add_quadratic_approx!(epi_cfg, jump_model, p2, p2_bounds, meta * "_diff")

    # Valid bounds for z ≈ x·y
    corners = (
        x_bounds.min * y_bounds.min,
        x_bounds.min * y_bounds.max,
        x_bounds.max * y_bounds.min,
        x_bounds.max * y_bounds.max,
    )
    z_lo = min(corners...)
    z_hi = max(corners...)

    z = JuMP.@variable(
        jump_model,
        base_name = "$(meta)_HybSProduct",
        lower_bound = z_lo,
        upper_bound = z_hi,
    )

    # Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y)
    lower_con = JuMP.@constraint(
        jump_model,
        z >= 0.5 * (zp1_r.result_expr - xsq - ysq),
    )
    # Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2)
    upper_con = JuMP.@constraint(
        jump_model,
        z <= 0.5 * (xsq + ysq - zp2_r.result_expr),
    )

    result = JuMP.AffExpr(0.0, z => 1.0)

    # Standard McCormick envelope cuts on the product variable
    mc_cons = nothing
    if config.add_mccormick
        mc_cons = Vector{JuMP.ConstraintRef}(undef, 4)
        mc_cons[1] = JuMP.@constraint(jump_model,
            z >= x_bounds.min * y + x * y_bounds.min - x_bounds.min * y_bounds.min)
        mc_cons[2] = JuMP.@constraint(jump_model,
            z >= x_bounds.max * y + x * y_bounds.max - x_bounds.max * y_bounds.max)
        mc_cons[3] = JuMP.@constraint(jump_model,
            z <= x_bounds.max * y + x * y_bounds.min - x_bounds.max * y_bounds.min)
        mc_cons[4] = JuMP.@constraint(jump_model,
            z <= x_bounds.min * y + x * y_bounds.max - x_bounds.min * y_bounds.max)
    end

    return (
        p1_expr = p1,
        p2_expr = p2,
        zp1_result = zp1_r,
        zp2_result = zp2_r,
        z_var = z,
        hybrid_cons = (lower_con, upper_con),
        result_expr = result,
        mc_cons = mc_cons,
    )
end

# --- Outer (container-aware) HybS bilinear approximation ---

"""
    add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps, xsq, ysq, x_var, y_var, x_bounds, y_bounds, meta)

Outer (container-aware) HybS bilinear approximation with pre-computed x² and y².

Creates named containers for all inner results — product variable, hybrid bound
constraints, result expression, sum/difference expressions for x±y, epigraph
containers for the (x+y)² and (x−y)² approximations, and (optionally) McCormick
envelope constraints — then calls the inner `_add_bilinear_approx!` per (name, time step).

# Arguments
- `config::HybSConfig`: HybS configuration
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `xsq`: container of pre-computed x² approximations indexed by (name, t)
- `ysq`: container of pre-computed y² approximations indexed by (name, t)
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_bounds::Vector{MinMax}`: per-name bounds for x
- `y_bounds::Vector{MinMax}`: per-name bounds for y
- `meta::String`: variable type identifier
"""
function add_bilinear_approx!(
    config::HybSConfig,
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

    z_var = add_variable_container!(
        container,
        BilinearProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    hybrid_cons = add_constraints_container!(
        container,
        HybSBoundConstraint(),
        C,
        names,
        1:2,
        time_steps;
        sparse = true,
        meta,
    )
    result_expr = add_expression_container!(
        container,
        BilinearProductExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    # Expression containers for p1 = x+y and p2 = x−y
    p1_expr_c = add_expression_container!(container, VariableSumExpression(), C, names, time_steps; meta)
    p2_expr_c = add_expression_container!(container, VariableDifferenceExpression(), C, names, time_steps; meta)

    # Epigraph containers for (x+y)² and (x−y)² approximations
    epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
    zp1_containers = _create_epi_containers!(epi_cfg, container, C, names, time_steps, meta * "_plus")
    zp2_containers = _create_epi_containers!(epi_cfg, container, C, names, time_steps, meta * "_diff")

    # McCormick envelope constraints (conditional)
    mc_con = nothing
    if config.add_mccormick
        mc_con = add_constraints_container!(container, McCormickConstraint(), C, names, 1:4, time_steps; meta)
    end

    for (i, name) in enumerate(names), t in time_steps
        r = _add_bilinear_approx!(
            config, jump_model,
            x_var[name, t], y_var[name, t],
            xsq[name, t], ysq[name, t],
            x_bounds[i], y_bounds[i], meta,
        )
        z_var[name, t] = r.z_var
        hybrid_cons[(name, 1, t)] = r.hybrid_cons[1]
        hybrid_cons[(name, 2, t)] = r.hybrid_cons[2]
        result_expr[name, t] = r.result_expr
        p1_expr_c[name, t] = r.p1_expr
        p2_expr_c[name, t] = r.p2_expr
        _store_epi_result!(epi_cfg, zp1_containers, name, i, t, r.zp1_result)
        _store_epi_result!(epi_cfg, zp2_containers, name, i, t, r.zp2_result)
        if !isnothing(mc_con)
            for k in 1:4
                mc_con[name, k, t] = r.mc_cons[k]
            end
        end
    end

    return result_expr
end

"""
    add_bilinear_approx!(config::HybSConfig, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Outer (container-aware) HybS bilinear approximation (non-precomputed).

Computes x² and y² quadratic approximations via `add_quadratic_approx!`, then
delegates to the precomputed form.

# Arguments
- `config::HybSConfig`: HybS configuration
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_bounds::Vector{MinMax}`: per-name bounds for x
- `y_bounds::Vector{MinMax}`: per-name bounds for y
- `meta::String`: variable type identifier
"""
function add_bilinear_approx!(
    config::HybSConfig,
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
