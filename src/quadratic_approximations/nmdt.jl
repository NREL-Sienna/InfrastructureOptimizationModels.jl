# NMDT (Normalized Multiparametric Disaggregation Technique) quadratic approximation of x².
# Normalizes x to [0,1], discretizes using L binary variables β₁,…,β_L plus a
# residual δ ∈ [0, 2^{−L}], then replaces each binary-continuous product β_i·xh
# with a McCormick-linearized auxiliary variable. Assembles the result via the
# separable identity x² = (lx·xh + x_min)². Optionally tightens with an epigraph
# lower bound on xh².
# NMDT Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation
# technique for global optimization of polynomial programming problems.

"""
Config for double-NMDT quadratic approximation.

# Fields
- `depth::Int`: number of binary discretization levels L
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound; 0 to disable (default 3×depth)
"""
struct DNMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
DNMDTQuadConfig(depth::Int) = DNMDTQuadConfig(depth, 3 * depth)

"""
Config for single-NMDT quadratic approximation.

# Fields
- `depth::Int`: number of binary discretization levels L
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound; 0 to disable (default 3×depth)
"""
struct NMDTQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
NMDTQuadConfig(depth::Int) = NMDTQuadConfig(depth, 3 * depth)

# ========================================================================================
# Inner functions — per-(name,t), container-unaware
# ========================================================================================

"""
    _add_quadratic_approx!(config::DNMDTQuadConfig, jump_model, x_disc, bounds, meta)

Inner (container-unaware) DNMDT quadratic approximation of x² from a pre-built
discretization for a single (name, t).

Constructs two binary-continuous products (β·xh and β·δ), assembles the DNMDT
result, and optionally tightens with an epigraph lower bound.

# Arguments
- `config::DNMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x_disc::NMDTDiscretization`: per-(name,t) discretization for x
- `bounds::MinMax`: `(min = x_min, max = x_max)` bounds for x
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple with fields:
- `result_expr::JuMP.AffExpr`: the DNMDT quadratic approximation of x²
- `bx_xh`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `bx_dx`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `dz_result`: named tuple from `_residual_product!` (z_var, mc_cons)
- `z1_expr::JuMP.AffExpr`: first DNMDT product estimate
- `z2_expr::JuMP.AffExpr`: second DNMDT product estimate
- `tight_result`: `nothing` or named tuple from `_tighten_lower_bounds!` (tight_con, epi_result)
"""
function _add_quadratic_approx!(
    config::DNMDTQuadConfig,
    jump_model::JuMP.Model,
    x_disc::NMDTDiscretization,
    bounds::MinMax,
    meta::String,
)
    tighten = config.epigraph_depth > 0
    bx_xh = _binary_continuous_product!(
        jump_model, x_disc, x_disc.norm_expr, 0.0, 1.0,
        config.depth, meta * "_bx_xh"; tighten,
    )
    bx_dx = _binary_continuous_product!(
        jump_model, x_disc, x_disc.delta_var, 0.0, 2.0^(-config.depth),
        config.depth, meta * "_bx_dx"; tighten,
    )

    asm = _assemble_dnmdt!(
        jump_model,
        bx_xh.result_expr, bx_dx.result_expr,
        bx_xh.result_expr, bx_dx.result_expr,
        x_disc, x_disc, bounds, bounds,
        config.depth, meta; tighten,
    )

    tight_result = nothing
    if tighten
        tight_result = _tighten_lower_bounds!(
            jump_model, asm.result_expr, x_disc.norm_expr,
            config.epigraph_depth, meta,
        )
    end

    return (
        result_expr = asm.result_expr,
        bx_xh = bx_xh,
        bx_dx = bx_dx,
        dz_result = asm.dz_result,
        z1_expr = asm.z1_expr,
        z2_expr = asm.z2_expr,
        tight_result = tight_result,
    )
end

"""
    _add_quadratic_approx!(config::DNMDTQuadConfig, jump_model, x, bounds, meta)

Inner (container-unaware) DNMDT quadratic approximation of x² from a raw variable
for a single (name, t).

Discretizes x via `_discretize!` then delegates to the precomputed overload.

# Arguments
- `config::DNMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x::JuMP.AbstractJuMPScalar`: the variable to square
- `bounds::MinMax`: `(min = x_min, max = x_max)` bounds for x
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple merging the precomputed DNMDT result with discretization fields:
- All fields from the precomputed DNMDT overload (result_expr, bx_xh, bx_dx, dz_result,
  z1_expr, z2_expr, tight_result)
- `disc::NMDTDiscretization`: the discretization created for x
- `disc_expr::JuMP.AffExpr`: discretization expression Σ 2^{−i}·β_i + δ
- `disc_con::JuMP.ConstraintRef`: constraint enforcing xh == disc_expr
"""
function _add_quadratic_approx!(
    config::DNMDTQuadConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    meta::String,
)
    disc_r = _discretize!(jump_model, x, bounds, config.depth, meta)
    quad = _add_quadratic_approx!(config, jump_model, disc_r.disc, bounds, meta)
    return merge(
        quad, (disc = disc_r.disc, disc_expr = disc_r.disc_expr, disc_con = disc_r.disc_con),
    )
end

"""
    _add_quadratic_approx!(config::NMDTQuadConfig, jump_model, x_disc, bounds, meta)

Inner (container-unaware) NMDT quadratic approximation of x² from a pre-built
discretization for a single (name, t).

Computes the binary-continuous product β·xh and residual product δ·xh, then
assembles x² via `_assemble_product`. Optionally tightens with an epigraph
lower bound.

# Arguments
- `config::NMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x_disc::NMDTDiscretization`: per-(name,t) discretization for x
- `bounds::MinMax`: `(min = x_min, max = x_max)` bounds for x
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple with fields:
- `result_expr::JuMP.AffExpr`: the NMDT quadratic approximation of x²
- `bx_y`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `dz`: named tuple from `_residual_product!` (z_var, mc_cons)
- `tight_result`: `nothing` or named tuple from `_tighten_lower_bounds!` (tight_con, epi_result)
"""
function _add_quadratic_approx!(
    config::NMDTQuadConfig,
    jump_model::JuMP.Model,
    x_disc::NMDTDiscretization,
    bounds::MinMax,
    meta::String,
)
    tighten = config.epigraph_depth > 0
    bx_y = _binary_continuous_product!(
        jump_model, x_disc, x_disc.norm_expr, 0.0, 1.0,
        config.depth, meta; tighten,
    )
    dz = _residual_product!(
        jump_model, x_disc, x_disc.norm_expr, 1.0,
        config.depth, meta; tighten,
    )

    result_expr = _assemble_product(
        [bx_y.result_expr], dz.z_var,
        x_disc.norm_expr, x_disc.norm_expr,
        bounds, bounds,
    )

    tight_result = nothing
    if tighten
        tight_result = _tighten_lower_bounds!(
            jump_model, result_expr, x_disc.norm_expr,
            config.epigraph_depth, meta,
        )
    end

    return (
        result_expr = result_expr,
        bx_y = bx_y,
        dz = dz,
        tight_result = tight_result,
    )
end

"""
    _add_quadratic_approx!(config::NMDTQuadConfig, jump_model, x, bounds, meta)

Inner (container-unaware) NMDT quadratic approximation of x² from a raw variable
for a single (name, t).

Discretizes x via `_discretize!` then delegates to the precomputed overload.

# Arguments
- `config::NMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x::JuMP.AbstractJuMPScalar`: the variable to square
- `bounds::MinMax`: `(min = x_min, max = x_max)` bounds for x
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple merging the precomputed NMDT result with discretization fields:
- All fields from the precomputed NMDT overload (result_expr, bx_y, dz, tight_result)
- `disc::NMDTDiscretization`: the discretization created for x
- `disc_expr::JuMP.AffExpr`: discretization expression Σ 2^{−i}·β_i + δ
- `disc_con::JuMP.ConstraintRef`: constraint enforcing xh == disc_expr
"""
function _add_quadratic_approx!(
    config::NMDTQuadConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    meta::String,
)
    disc_r = _discretize!(jump_model, x, bounds, config.depth, meta)
    quad = _add_quadratic_approx!(config, jump_model, disc_r.disc, bounds, meta)
    return merge(
        quad, (disc = disc_r.disc, disc_expr = disc_r.disc_expr, disc_con = disc_r.disc_con),
    )
end

# ========================================================================================
# Outer functions — container-aware, loop over (name, t)
# ========================================================================================

"""
    add_quadratic_approx!(config::DNMDTQuadConfig, container, C, names, time_steps, x_discs, bounds, meta)

Outer (container-aware) DNMDT quadratic approximation of x² from pre-built
per-(name,t) discretizations.

Creates containers for all intermediate JuMP objects (binary-continuous product
variables/constraints/expressions, residual product variable, tightening
constraint, and epigraph containers), loops over all (name, t) calling the
inner precomputed DNMDT method, and populates every container.

# Arguments
- `config::DNMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_discs::AbstractDict{Tuple{String, Int}}`: per-(name,t) discretizations
- `bounds::Vector{MinMax}`: per-name bounds `[(min=x_min, max=x_max), ...]`
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_quadratic_approx!(
    config::DNMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_discs::AbstractDict{Tuple{String, Int}},
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    depth = config.depth
    tighten = config.epigraph_depth > 0

    result_expr = add_expression_container!(
        container, QuadraticExpression(), C, names, time_steps; meta,
    )
    bx_xh_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_xh",
    )
    bx_dx_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_dx",
    )
    bx_xh_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_xh", sparse = true,
    )
    bx_dx_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_dx", sparse = true,
    )
    bx_xh_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_xh",
    )
    bx_dx_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_dx",
    )
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    tight_con_c = nothing
    epi_containers = nothing
    epi_cfg = nothing
    if tighten
        tight_con_c = add_constraints_container!(
            container, NMDTTightenConstraint(), C, names, time_steps; meta,
        )
        epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
        epi_containers = _create_epi_containers!(
            epi_cfg, container, C, names, time_steps, meta * "_epi",
        )
    end

    for (i, name) in enumerate(names), t in time_steps
        r = _add_quadratic_approx!(
            config, jump_model, x_discs[(name, t)], bounds[i], meta,
        )
        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_xh_u[name, j, t] = r.bx_xh.u_var[j]
        end
        bx_xh_expr[name, t] = r.bx_xh.result_expr
        for j in 1:depth
            if !tighten
                bx_xh_mc[name, j, 1, t] = r.bx_xh.mc_cons[j, 1]
                bx_xh_mc[name, j, 2, t] = r.bx_xh.mc_cons[j, 2]
            end
            bx_xh_mc[name, j, 3, t] = r.bx_xh.mc_cons[j, 3]
            bx_xh_mc[name, j, 4, t] = r.bx_xh.mc_cons[j, 4]
        end

        for j in 1:depth
            bx_dx_u[name, j, t] = r.bx_dx.u_var[j]
        end
        bx_dx_expr[name, t] = r.bx_dx.result_expr
        for j in 1:depth
            if !tighten
                bx_dx_mc[name, j, 1, t] = r.bx_dx.mc_cons[j, 1]
                bx_dx_mc[name, j, 2, t] = r.bx_dx.mc_cons[j, 2]
            end
            bx_dx_mc[name, j, 3, t] = r.bx_dx.mc_cons[j, 3]
            bx_dx_mc[name, j, 4, t] = r.bx_dx.mc_cons[j, 4]
        end

        dz_var_c[name, t] = r.dz_result.z_var

        if !isnothing(tight_con_c)
            tight_con_c[name, t] = r.tight_result.tight_con
            _store_epi_result!(
                epi_cfg, epi_containers, name, i, t, r.tight_result.epi_result,
            )
        end
    end

    return result_expr
end

"""
    add_quadratic_approx!(config::DNMDTQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Outer (container-aware) DNMDT quadratic approximation of x² from raw variables.

Creates all optimization containers upfront (discretization variables/expressions/
constraints, binary-continuous product variables/constraints/expressions, residual
product variable, tightening constraint, and epigraph containers), then calls the
inner non-precomputed DNMDT method per (name, t) to populate them.

# Arguments
- `config::DNMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name bounds `[(min=x_min, max=x_max), ...]`
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_quadratic_approx!(
    config::DNMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    depth = config.depth
    tighten = config.epigraph_depth > 0

    beta_var = add_variable_container!(
        container, NMDTBinaryVariable(), C, names, 1:depth, time_steps; meta,
    )
    delta_var = add_variable_container!(
        container, NMDTResidualVariable(), C, names, time_steps; meta,
    )
    result_expr = add_expression_container!(
        container, QuadraticExpression(), C, names, time_steps; meta,
    )
    disc_expr_c = add_expression_container!(
        container, NMDTDiscretizationExpression(), C, names, time_steps; meta,
    )
    disc_con_c = add_constraints_container!(
        container, NMDTEDiscretizationConstraint(), C, names, time_steps; meta,
    )
    bx_xh_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_xh",
    )
    bx_dx_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_dx",
    )
    bx_xh_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_xh", sparse = true,
    )
    bx_dx_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_dx", sparse = true,
    )
    bx_xh_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_xh",
    )
    bx_dx_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_dx",
    )
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    tight_con_c = nothing
    epi_containers = nothing
    epi_cfg = nothing
    if tighten
        tight_con_c = add_constraints_container!(
            container, NMDTTightenConstraint(), C, names, time_steps; meta,
        )
        epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
        epi_containers = _create_epi_containers!(
            epi_cfg, container, C, names, time_steps, meta * "_epi",
        )
    end

    for (i, name) in enumerate(names), t in time_steps
        r = _add_quadratic_approx!(
            config, jump_model, x_var[name, t], bounds[i], meta,
        )
        for j in 1:depth
            beta_var[name, j, t] = r.disc.beta_var[j]
        end
        delta_var[name, t] = r.disc.delta_var
        disc_expr_c[name, t] = r.disc_expr
        disc_con_c[name, t] = r.disc_con
        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_xh_u[name, j, t] = r.bx_xh.u_var[j]
        end
        bx_xh_expr[name, t] = r.bx_xh.result_expr
        for j in 1:depth
            if !tighten
                bx_xh_mc[name, j, 1, t] = r.bx_xh.mc_cons[j, 1]
                bx_xh_mc[name, j, 2, t] = r.bx_xh.mc_cons[j, 2]
            end
            bx_xh_mc[name, j, 3, t] = r.bx_xh.mc_cons[j, 3]
            bx_xh_mc[name, j, 4, t] = r.bx_xh.mc_cons[j, 4]
        end

        for j in 1:depth
            bx_dx_u[name, j, t] = r.bx_dx.u_var[j]
        end
        bx_dx_expr[name, t] = r.bx_dx.result_expr
        for j in 1:depth
            if !tighten
                bx_dx_mc[name, j, 1, t] = r.bx_dx.mc_cons[j, 1]
                bx_dx_mc[name, j, 2, t] = r.bx_dx.mc_cons[j, 2]
            end
            bx_dx_mc[name, j, 3, t] = r.bx_dx.mc_cons[j, 3]
            bx_dx_mc[name, j, 4, t] = r.bx_dx.mc_cons[j, 4]
        end

        dz_var_c[name, t] = r.dz_result.z_var

        if !isnothing(tight_con_c)
            tight_con_c[name, t] = r.tight_result.tight_con
            _store_epi_result!(
                epi_cfg, epi_containers, name, i, t, r.tight_result.epi_result,
            )
        end
    end

    return result_expr
end

"""
    add_quadratic_approx!(config::NMDTQuadConfig, container, C, names, time_steps, x_discs, bounds, meta)

Outer (container-aware) NMDT quadratic approximation of x² from pre-built
per-(name,t) discretizations.

Creates containers for all intermediate JuMP objects (binary-continuous product
variables/constraints/expressions, residual product variable, tightening
constraint, and epigraph containers), loops over all (name, t) calling the
inner precomputed NMDT method, and populates every container.

# Arguments
- `config::NMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_discs::AbstractDict{Tuple{String, Int}}`: per-(name,t) discretizations
- `bounds::Vector{MinMax}`: per-name bounds `[(min=x_min, max=x_max), ...]`
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_quadratic_approx!(
    config::NMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_discs::AbstractDict{Tuple{String, Int}},
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    depth = config.depth
    tighten = config.epigraph_depth > 0

    result_expr = add_expression_container!(
        container, QuadraticExpression(), C, names, time_steps; meta,
    )
    bx_y_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta,
    )
    bx_y_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta, sparse = true,
    )
    bx_y_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps; meta,
    )
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    tight_con_c = nothing
    epi_containers = nothing
    epi_cfg = nothing
    if tighten
        tight_con_c = add_constraints_container!(
            container, NMDTTightenConstraint(), C, names, time_steps; meta,
        )
        epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
        epi_containers = _create_epi_containers!(
            epi_cfg, container, C, names, time_steps, meta * "_epi",
        )
    end

    for (i, name) in enumerate(names), t in time_steps
        r = _add_quadratic_approx!(
            config, jump_model, x_discs[(name, t)], bounds[i], meta,
        )
        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_y_u[name, j, t] = r.bx_y.u_var[j]
        end
        bx_y_expr[name, t] = r.bx_y.result_expr
        for j in 1:depth
            if !tighten
                bx_y_mc[name, j, 1, t] = r.bx_y.mc_cons[j, 1]
                bx_y_mc[name, j, 2, t] = r.bx_y.mc_cons[j, 2]
            end
            bx_y_mc[name, j, 3, t] = r.bx_y.mc_cons[j, 3]
            bx_y_mc[name, j, 4, t] = r.bx_y.mc_cons[j, 4]
        end

        dz_var_c[name, t] = r.dz.z_var

        if !isnothing(tight_con_c)
            tight_con_c[name, t] = r.tight_result.tight_con
            _store_epi_result!(
                epi_cfg, epi_containers, name, i, t, r.tight_result.epi_result,
            )
        end
    end

    return result_expr
end

"""
    add_quadratic_approx!(config::NMDTQuadConfig, container, C, names, time_steps, x_var, bounds, meta)

Outer (container-aware) NMDT quadratic approximation of x² from raw variables.

Creates all optimization containers upfront (discretization variables/expressions/
constraints, binary-continuous product variables/constraints/expressions, residual
product variable, tightening constraint, and epigraph containers), then calls the
inner non-precomputed NMDT method per (name, t) to populate them.

# Arguments
- `config::NMDTQuadConfig`: configuration with `depth` and `epigraph_depth`
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `bounds::Vector{MinMax}`: per-name bounds `[(min=x_min, max=x_max), ...]`
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_quadratic_approx!(
    config::NMDTQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    depth = config.depth
    tighten = config.epigraph_depth > 0

    beta_var = add_variable_container!(
        container, NMDTBinaryVariable(), C, names, 1:depth, time_steps; meta,
    )
    delta_var = add_variable_container!(
        container, NMDTResidualVariable(), C, names, time_steps; meta,
    )
    result_expr = add_expression_container!(
        container, QuadraticExpression(), C, names, time_steps; meta,
    )
    disc_expr_c = add_expression_container!(
        container, NMDTDiscretizationExpression(), C, names, time_steps; meta,
    )
    disc_con_c = add_constraints_container!(
        container, NMDTEDiscretizationConstraint(), C, names, time_steps; meta,
    )
    bx_y_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta,
    )
    bx_y_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta, sparse = true,
    )
    bx_y_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps; meta,
    )
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    tight_con_c = nothing
    epi_containers = nothing
    epi_cfg = nothing
    if tighten
        tight_con_c = add_constraints_container!(
            container, NMDTTightenConstraint(), C, names, time_steps; meta,
        )
        epi_cfg = EpigraphQuadConfig(config.epigraph_depth)
        epi_containers = _create_epi_containers!(
            epi_cfg, container, C, names, time_steps, meta * "_epi",
        )
    end

    for (i, name) in enumerate(names), t in time_steps
        r = _add_quadratic_approx!(
            config, jump_model, x_var[name, t], bounds[i], meta,
        )
        for j in 1:depth
            beta_var[name, j, t] = r.disc.beta_var[j]
        end
        delta_var[name, t] = r.disc.delta_var
        disc_expr_c[name, t] = r.disc_expr
        disc_con_c[name, t] = r.disc_con
        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_y_u[name, j, t] = r.bx_y.u_var[j]
        end
        bx_y_expr[name, t] = r.bx_y.result_expr
        for j in 1:depth
            if !tighten
                bx_y_mc[name, j, 1, t] = r.bx_y.mc_cons[j, 1]
                bx_y_mc[name, j, 2, t] = r.bx_y.mc_cons[j, 2]
            end
            bx_y_mc[name, j, 3, t] = r.bx_y.mc_cons[j, 3]
            bx_y_mc[name, j, 4, t] = r.bx_y.mc_cons[j, 4]
        end

        dz_var_c[name, t] = r.dz.z_var

        if !isnothing(tight_con_c)
            tight_con_c[name, t] = r.tight_result.tight_con
            _store_epi_result!(
                epi_cfg, epi_containers, name, i, t, r.tight_result.epi_result,
            )
        end
    end

    return result_expr
end
