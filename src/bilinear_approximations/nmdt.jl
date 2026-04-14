# DNMDT (Double Normalized Multiparametric Disaggregation Technique) bilinear approximation of x·y.
# Independently discretizes both x and y, forms four cross binary-continuous products, then
# combines two NMDT estimates with a convex weighting λ (default 0.5). Reduces to the NMDT
# formulation when applied to x·x (quadratic case).
# Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation technique for global
# optimization of polynomial programming problems.

"""
Config for double-NMDT bilinear approximation (discretizes both x and y).

# Fields
- `depth::Int`: number of binary discretization levels L for both x and y
"""
struct DNMDTBilinearConfig <: BilinearApproxConfig
    depth::Int
end

"""
Config for single-NMDT bilinear approximation (discretizes x only).

# Fields
- `depth::Int`: number of binary discretization levels L for x
"""
struct NMDTBilinearConfig <: BilinearApproxConfig
    depth::Int
end

# ═══════════════════════════════════════════════════════════════════════════════
# Inner functions — operate on a single (name, t) using JuMP.Model directly
# ═══════════════════════════════════════════════════════════════════════════════

# --- DNMDT inner functions ---

"""
    _add_bilinear_approx!(config::DNMDTBilinearConfig, jump_model, x_disc, y_disc, x_bounds, y_bounds, meta)

Inner function: approximate x·y for a single (name, t) using the DNMDT method from
pre-built discretizations.

Constructs all four cross binary-continuous products (β_x·ŷ, β_y·δx, β_y·x̂, β_x·δy)
then delegates to `_assemble_dnmdt!`.

# Arguments
- `config::DNMDTBilinearConfig`: DNMDT configuration
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `y_disc::NMDTDiscretization`: pre-built discretization for y
- `x_bounds::MinMax`: `(min, max)` bounds for x
- `y_bounds::MinMax`: `(min, max)` bounds for y
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple with fields:
- `result_expr::JuMP.AffExpr`: the DNMDT bilinear approximation of x·y
- `bx_yh`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `by_dx`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `by_xh`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `bx_dy`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `dz_result`: named tuple from `_residual_product!` (z_var, mc_cons)
- `z1_expr::JuMP.AffExpr`: first NMDT product estimate
- `z2_expr::JuMP.AffExpr`: second NMDT product estimate
"""
function _add_bilinear_approx!(
    config::DNMDTBilinearConfig,
    jump_model::JuMP.Model,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_bounds::MinMax,
    y_bounds::MinMax,
    meta::String,
)
    bx_yh = _binary_continuous_product!(
        jump_model, x_disc, y_disc.norm_expr,
        0.0, 1.0, config.depth, meta * "_bx_yh",
    )
    by_dx = _binary_continuous_product!(
        jump_model, y_disc, x_disc.delta_var,
        0.0, 2.0^(-config.depth), config.depth, meta * "_by_dx",
    )
    by_xh = _binary_continuous_product!(
        jump_model, y_disc, x_disc.norm_expr,
        0.0, 1.0, config.depth, meta * "_by_xh",
    )
    bx_dy = _binary_continuous_product!(
        jump_model, x_disc, y_disc.delta_var,
        0.0, 2.0^(-config.depth), config.depth, meta * "_bx_dy",
    )

    asm = _assemble_dnmdt!(
        jump_model,
        bx_yh.result_expr, by_dx.result_expr,
        by_xh.result_expr, bx_dy.result_expr,
        x_disc, y_disc, x_bounds, y_bounds,
        config.depth, meta,
    )

    return (
        result_expr = asm.result_expr,
        bx_yh = bx_yh,
        by_dx = by_dx,
        by_xh = by_xh,
        bx_dy = bx_dy,
        dz_result = asm.dz_result,
        z1_expr = asm.z1_expr,
        z2_expr = asm.z2_expr,
    )
end

"""
    _add_bilinear_approx!(config::DNMDTBilinearConfig, jump_model, x, y, x_bounds, y_bounds, meta)

Inner function: approximate x·y for a single (name, t) using the DNMDT method from
raw variable inputs.

Discretizes both x and y independently via `_discretize!` then delegates to the
pre-discretized inner overload.

# Arguments
- `config::DNMDTBilinearConfig`: DNMDT configuration
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x::JuMP.AbstractJuMPScalar`: single x variable
- `y::JuMP.AbstractJuMPScalar`: single y variable
- `x_bounds::MinMax`: `(min, max)` bounds for x
- `y_bounds::MinMax`: `(min, max)` bounds for y
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple merging the precomputed DNMDT result with discretization fields:
- All fields from the precomputed DNMDT overload (result_expr, bx_yh, by_dx, by_xh,
  bx_dy, dz_result, z1_expr, z2_expr)
- `x_disc::NMDTDiscretization`: the discretization created for x
- `x_disc_expr::JuMP.AffExpr`: discretization expression for x
- `x_disc_con::JuMP.ConstraintRef`: constraint enforcing xh == disc_expr for x
- `y_disc::NMDTDiscretization`: the discretization created for y
- `y_disc_expr::JuMP.AffExpr`: discretization expression for y
- `y_disc_con::JuMP.ConstraintRef`: constraint enforcing yh == disc_expr for y
"""
function _add_bilinear_approx!(
    config::DNMDTBilinearConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    x_bounds::MinMax,
    y_bounds::MinMax,
    meta::String,
)
    x_disc_r = _discretize!(jump_model, x, x_bounds, config.depth, meta * "_x")
    y_disc_r = _discretize!(jump_model, y, y_bounds, config.depth, meta * "_y")
    r = _add_bilinear_approx!(
        config, jump_model, x_disc_r.disc, y_disc_r.disc,
        x_bounds, y_bounds, meta,
    )
    return merge(r, (
        x_disc = x_disc_r.disc, x_disc_expr = x_disc_r.disc_expr, x_disc_con = x_disc_r.disc_con,
        y_disc = y_disc_r.disc, y_disc_expr = y_disc_r.disc_expr, y_disc_con = y_disc_r.disc_con,
    ))
end

# --- NMDT inner functions ---

"""
    _add_bilinear_approx!(config::NMDTBilinearConfig, jump_model, x_disc, yh, x_bounds, y_bounds, meta)

Inner function: approximate x·y for a single (name, t) using the NMDT method from a
pre-built x discretization and normalized y.

Computes binary-continuous product β_x·ŷ and residual product δ_x·ŷ, then assembles
the bilinear product via `_assemble_product`.

# Arguments
- `config::NMDTBilinearConfig`: NMDT configuration
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `yh::JuMP.AffExpr`: normalized expression ŷ = (y − y_min)/(y_max − y_min)
- `x_bounds::MinMax`: `(min, max)` bounds for x
- `y_bounds::MinMax`: `(min, max)` bounds for y
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple with fields:
- `result_expr::JuMP.AffExpr`: the NMDT bilinear approximation of x·y
- `bx_y`: named tuple from `_binary_continuous_product!` (u_var, mc_cons, result_expr)
- `dz`: named tuple from `_residual_product!` (z_var, mc_cons)
"""
function _add_bilinear_approx!(
    config::NMDTBilinearConfig,
    jump_model::JuMP.Model,
    x_disc::NMDTDiscretization,
    yh::JuMP.AffExpr,
    x_bounds::MinMax,
    y_bounds::MinMax,
    meta::String,
)
    bx_y = _binary_continuous_product!(
        jump_model, x_disc, yh, 0.0, 1.0,
        config.depth, meta,
    )
    dz = _residual_product!(
        jump_model, x_disc, yh, 1.0,
        config.depth, meta,
    )
    result = _assemble_product(
        [bx_y.result_expr], dz.z_var,
        x_disc.norm_expr, yh,
        x_bounds, y_bounds,
    )
    return (
        result_expr = result,
        bx_y = bx_y,
        dz = dz,
    )
end

"""
    _add_bilinear_approx!(config::NMDTBilinearConfig, jump_model, x, y, x_bounds, y_bounds, meta)

Inner function: approximate x·y for a single (name, t) using the NMDT method from
raw variable inputs.

Discretizes x via `_discretize!` and normalizes y via `_normed_variable`, then
delegates to the pre-discretized inner overload.

# Arguments
- `config::NMDTBilinearConfig`: NMDT configuration
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x::JuMP.AbstractJuMPScalar`: single x variable
- `y::JuMP.AbstractJuMPScalar`: single y variable
- `x_bounds::MinMax`: `(min, max)` bounds for x
- `y_bounds::MinMax`: `(min, max)` bounds for y
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple merging the precomputed NMDT result with discretization fields:
- All fields from the precomputed NMDT overload (result_expr, bx_y, dz)
- `x_disc::NMDTDiscretization`: the discretization created for x
- `x_disc_expr::JuMP.AffExpr`: discretization expression for x
- `x_disc_con::JuMP.ConstraintRef`: constraint enforcing xh == disc_expr for x
- `yh::JuMP.AffExpr`: normalized y expression
"""
function _add_bilinear_approx!(
    config::NMDTBilinearConfig,
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    x_bounds::MinMax,
    y_bounds::MinMax,
    meta::String,
)
    x_disc_r = _discretize!(jump_model, x, x_bounds, config.depth, meta * "_x")
    yh = _normed_variable(y, y_bounds)
    r = _add_bilinear_approx!(
        config, jump_model, x_disc_r.disc, yh,
        x_bounds, y_bounds, meta,
    )
    return merge(r, (
        x_disc = x_disc_r.disc, x_disc_expr = x_disc_r.disc_expr, x_disc_con = x_disc_r.disc_con,
        yh = yh,
    ))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Outer functions — loop over (name, t), manage containers
# ═══════════════════════════════════════════════════════════════════════════════

# --- DNMDT outer functions ---

"""
    add_bilinear_approx!(config::DNMDTBilinearConfig, container, C, names, time_steps, x_discs, y_discs, x_bounds, y_bounds, meta)

Outer (container-aware) DNMDT bilinear approximation of x·y from pre-built
per-(name,t) discretizations.

Creates containers for all intermediate JuMP objects (four binary-continuous product
variables/constraints/expressions and residual product variable), loops over all
(name, t) calling the inner precomputed DNMDT method, and populates every container.

# Arguments
- `config::DNMDTBilinearConfig`: DNMDT configuration
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_discs::AbstractDict{Tuple{String, Int}}`: pre-built discretizations for x
- `y_discs::AbstractDict{Tuple{String, Int}}`: pre-built discretizations for y
- `x_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for x
- `y_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_bilinear_approx!(
    config::DNMDTBilinearConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_discs::AbstractDict{Tuple{String, Int}},
    y_discs::AbstractDict{Tuple{String, Int}},
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    depth = config.depth

    result_expr = add_expression_container!(
        container, BilinearProductExpression(), C, names, time_steps; meta,
    )

    # Binary-continuous product containers (4 cross products)
    bx_yh_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_yh",
    )
    by_dx_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_by_dx",
    )
    by_xh_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_by_xh",
    )
    bx_dy_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_dy",
    )

    bx_yh_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_yh", sparse = true,
    )
    by_dx_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_by_dx", sparse = true,
    )
    by_xh_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_by_xh", sparse = true,
    )
    bx_dy_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_dy", sparse = true,
    )

    bx_yh_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_yh",
    )
    by_dx_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_by_dx",
    )
    by_xh_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_by_xh",
    )
    bx_dy_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_dy",
    )

    # Residual product container
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        r = _add_bilinear_approx!(
            config, jump_model, x_discs[(name, t)], y_discs[(name, t)],
            x_bounds[i], y_bounds[i], meta,
        )
        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_yh_u[name, j, t] = r.bx_yh.u_var[j]
            bx_yh_mc[name, j, 1, t] = r.bx_yh.mc_cons[j, 1]
            bx_yh_mc[name, j, 2, t] = r.bx_yh.mc_cons[j, 2]
            bx_yh_mc[name, j, 3, t] = r.bx_yh.mc_cons[j, 3]
            bx_yh_mc[name, j, 4, t] = r.bx_yh.mc_cons[j, 4]
        end
        bx_yh_expr[name, t] = r.bx_yh.result_expr

        for j in 1:depth
            by_dx_u[name, j, t] = r.by_dx.u_var[j]
            by_dx_mc[name, j, 1, t] = r.by_dx.mc_cons[j, 1]
            by_dx_mc[name, j, 2, t] = r.by_dx.mc_cons[j, 2]
            by_dx_mc[name, j, 3, t] = r.by_dx.mc_cons[j, 3]
            by_dx_mc[name, j, 4, t] = r.by_dx.mc_cons[j, 4]
        end
        by_dx_expr[name, t] = r.by_dx.result_expr

        for j in 1:depth
            by_xh_u[name, j, t] = r.by_xh.u_var[j]
            by_xh_mc[name, j, 1, t] = r.by_xh.mc_cons[j, 1]
            by_xh_mc[name, j, 2, t] = r.by_xh.mc_cons[j, 2]
            by_xh_mc[name, j, 3, t] = r.by_xh.mc_cons[j, 3]
            by_xh_mc[name, j, 4, t] = r.by_xh.mc_cons[j, 4]
        end
        by_xh_expr[name, t] = r.by_xh.result_expr

        for j in 1:depth
            bx_dy_u[name, j, t] = r.bx_dy.u_var[j]
            bx_dy_mc[name, j, 1, t] = r.bx_dy.mc_cons[j, 1]
            bx_dy_mc[name, j, 2, t] = r.bx_dy.mc_cons[j, 2]
            bx_dy_mc[name, j, 3, t] = r.bx_dy.mc_cons[j, 3]
            bx_dy_mc[name, j, 4, t] = r.bx_dy.mc_cons[j, 4]
        end
        bx_dy_expr[name, t] = r.bx_dy.result_expr

        dz_var_c[name, t] = r.dz_result.z_var
    end

    return result_expr
end

"""
    add_bilinear_approx!(config::DNMDTBilinearConfig, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Outer (container-aware) DNMDT bilinear approximation of x·y from raw variables.

Creates all optimization containers upfront (discretization variables/expressions/
constraints for both x and y, four binary-continuous product variables/constraints/
expressions, and residual product variable), then calls the inner non-precomputed
DNMDT method per (name, t) to populate them.

# Arguments
- `config::DNMDTBilinearConfig`: DNMDT configuration
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
    config::DNMDTBilinearConfig,
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
    jump_model = get_jump_model(container)
    depth = config.depth

    # Discretization containers for x and y
    x_beta_var = add_variable_container!(
        container, NMDTBinaryVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_x",
    )
    x_delta_var = add_variable_container!(
        container, NMDTResidualVariable(), C, names, time_steps;
        meta = meta * "_x",
    )
    x_disc_expr_c = add_expression_container!(
        container, NMDTDiscretizationExpression(), C, names, time_steps;
        meta = meta * "_x",
    )
    x_disc_con_c = add_constraints_container!(
        container, NMDTEDiscretizationConstraint(), C, names, time_steps;
        meta = meta * "_x",
    )
    y_beta_var = add_variable_container!(
        container, NMDTBinaryVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_y",
    )
    y_delta_var = add_variable_container!(
        container, NMDTResidualVariable(), C, names, time_steps;
        meta = meta * "_y",
    )
    y_disc_expr_c = add_expression_container!(
        container, NMDTDiscretizationExpression(), C, names, time_steps;
        meta = meta * "_y",
    )
    y_disc_con_c = add_constraints_container!(
        container, NMDTEDiscretizationConstraint(), C, names, time_steps;
        meta = meta * "_y",
    )

    result_expr = add_expression_container!(
        container, BilinearProductExpression(), C, names, time_steps; meta,
    )

    # Binary-continuous product containers (4 cross products)
    bx_yh_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_yh",
    )
    by_dx_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_by_dx",
    )
    by_xh_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_by_xh",
    )
    bx_dy_u = add_variable_container!(
        container, NMDTBinaryContinuousProductVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_bx_dy",
    )

    bx_yh_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_yh", sparse = true,
    )
    by_dx_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_by_dx", sparse = true,
    )
    by_xh_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_by_xh", sparse = true,
    )
    bx_dy_mc = add_constraints_container!(
        container, NMDTBinaryContinuousProductConstraint(), C, names, 1:depth, 1:4, time_steps;
        meta = meta * "_bx_dy", sparse = true,
    )

    bx_yh_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_yh",
    )
    by_dx_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_by_dx",
    )
    by_xh_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_by_xh",
    )
    bx_dy_expr = add_expression_container!(
        container, NMDTBinaryContinuousProductExpression(), C, names, time_steps;
        meta = meta * "_bx_dy",
    )

    # Residual product container
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        r = _add_bilinear_approx!(
            config, jump_model, x_var[name, t], y_var[name, t],
            x_bounds[i], y_bounds[i], meta,
        )
        for j in 1:depth
            x_beta_var[name, j, t] = r.x_disc.beta_var[j]
        end
        x_delta_var[name, t] = r.x_disc.delta_var
        x_disc_expr_c[name, t] = r.x_disc_expr
        x_disc_con_c[name, t] = r.x_disc_con
        for j in 1:depth
            y_beta_var[name, j, t] = r.y_disc.beta_var[j]
        end
        y_delta_var[name, t] = r.y_disc.delta_var
        y_disc_expr_c[name, t] = r.y_disc_expr
        y_disc_con_c[name, t] = r.y_disc_con

        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_yh_u[name, j, t] = r.bx_yh.u_var[j]
            bx_yh_mc[name, j, 1, t] = r.bx_yh.mc_cons[j, 1]
            bx_yh_mc[name, j, 2, t] = r.bx_yh.mc_cons[j, 2]
            bx_yh_mc[name, j, 3, t] = r.bx_yh.mc_cons[j, 3]
            bx_yh_mc[name, j, 4, t] = r.bx_yh.mc_cons[j, 4]
        end
        bx_yh_expr[name, t] = r.bx_yh.result_expr

        for j in 1:depth
            by_dx_u[name, j, t] = r.by_dx.u_var[j]
            by_dx_mc[name, j, 1, t] = r.by_dx.mc_cons[j, 1]
            by_dx_mc[name, j, 2, t] = r.by_dx.mc_cons[j, 2]
            by_dx_mc[name, j, 3, t] = r.by_dx.mc_cons[j, 3]
            by_dx_mc[name, j, 4, t] = r.by_dx.mc_cons[j, 4]
        end
        by_dx_expr[name, t] = r.by_dx.result_expr

        for j in 1:depth
            by_xh_u[name, j, t] = r.by_xh.u_var[j]
            by_xh_mc[name, j, 1, t] = r.by_xh.mc_cons[j, 1]
            by_xh_mc[name, j, 2, t] = r.by_xh.mc_cons[j, 2]
            by_xh_mc[name, j, 3, t] = r.by_xh.mc_cons[j, 3]
            by_xh_mc[name, j, 4, t] = r.by_xh.mc_cons[j, 4]
        end
        by_xh_expr[name, t] = r.by_xh.result_expr

        for j in 1:depth
            bx_dy_u[name, j, t] = r.bx_dy.u_var[j]
            bx_dy_mc[name, j, 1, t] = r.bx_dy.mc_cons[j, 1]
            bx_dy_mc[name, j, 2, t] = r.bx_dy.mc_cons[j, 2]
            bx_dy_mc[name, j, 3, t] = r.bx_dy.mc_cons[j, 3]
            bx_dy_mc[name, j, 4, t] = r.bx_dy.mc_cons[j, 4]
        end
        bx_dy_expr[name, t] = r.bx_dy.result_expr

        dz_var_c[name, t] = r.dz_result.z_var
    end

    return result_expr
end

# --- NMDT outer functions ---

"""
    add_bilinear_approx!(config::NMDTBilinearConfig, container, C, names, time_steps, x_discs, yh_exprs, x_bounds, y_bounds, meta)

Outer (container-aware) NMDT bilinear approximation of x·y from pre-built
per-(name,t) x discretizations and normalized y expressions.

Creates containers for all intermediate JuMP objects (binary-continuous product
variables/constraints/expressions and residual product variable), loops over all
(name, t) calling the inner precomputed NMDT method, and populates every container.

# Arguments
- `config::NMDTBilinearConfig`: NMDT configuration
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_discs::AbstractDict{Tuple{String, Int}}`: pre-built discretizations for x
- `yh_exprs::AbstractDict{Tuple{String, Int}}`: normalized y expressions
- `x_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for x
- `y_bounds::Vector{MinMax}`: per-name `(min, max)` bounds for y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function add_bilinear_approx!(
    config::NMDTBilinearConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_discs::AbstractDict{Tuple{String, Int}},
    yh_exprs::AbstractDict{Tuple{String, Int}},
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    depth = config.depth

    result_expr = add_expression_container!(
        container, BilinearProductExpression(), C, names, time_steps; meta,
    )

    # Binary-continuous product containers
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

    # Residual product container
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        r = _add_bilinear_approx!(
            config, jump_model, x_discs[(name, t)], yh_exprs[(name, t)],
            x_bounds[i], y_bounds[i], meta,
        )
        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_y_u[name, j, t] = r.bx_y.u_var[j]
            bx_y_mc[name, j, 1, t] = r.bx_y.mc_cons[j, 1]
            bx_y_mc[name, j, 2, t] = r.bx_y.mc_cons[j, 2]
            bx_y_mc[name, j, 3, t] = r.bx_y.mc_cons[j, 3]
            bx_y_mc[name, j, 4, t] = r.bx_y.mc_cons[j, 4]
        end
        bx_y_expr[name, t] = r.bx_y.result_expr

        dz_var_c[name, t] = r.dz.z_var
    end

    return result_expr
end

"""
    add_bilinear_approx!(config::NMDTBilinearConfig, container, C, names, time_steps, x_var, y_var, x_bounds, y_bounds, meta)

Outer (container-aware) NMDT bilinear approximation of x·y from raw variables.

Creates all optimization containers upfront (x discretization variables/expressions/
constraints, binary-continuous product variables/constraints/expressions, and residual
product variable), then calls the inner non-precomputed NMDT method per (name, t) to
populate them. Note: y is only normalized (not discretized), so no y discretization
containers are needed.

# Arguments
- `config::NMDTBilinearConfig`: NMDT configuration
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
    config::NMDTBilinearConfig,
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
    jump_model = get_jump_model(container)
    depth = config.depth

    # Discretization containers for x
    beta_var = add_variable_container!(
        container, NMDTBinaryVariable(), C, names, 1:depth, time_steps;
        meta = meta * "_x",
    )
    delta_var = add_variable_container!(
        container, NMDTResidualVariable(), C, names, time_steps;
        meta = meta * "_x",
    )
    disc_expr_c = add_expression_container!(
        container, NMDTDiscretizationExpression(), C, names, time_steps;
        meta = meta * "_x",
    )
    disc_con_c = add_constraints_container!(
        container, NMDTEDiscretizationConstraint(), C, names, time_steps;
        meta = meta * "_x",
    )

    result_expr = add_expression_container!(
        container, BilinearProductExpression(), C, names, time_steps; meta,
    )

    # Binary-continuous product containers
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

    # Residual product container
    dz_var_c = add_variable_container!(
        container, NMDTResidualProductVariable(), C, names, time_steps; meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        r = _add_bilinear_approx!(
            config, jump_model, x_var[name, t], y_var[name, t],
            x_bounds[i], y_bounds[i], meta,
        )
        for j in 1:depth
            beta_var[name, j, t] = r.x_disc.beta_var[j]
        end
        delta_var[name, t] = r.x_disc.delta_var
        disc_expr_c[name, t] = r.x_disc_expr
        disc_con_c[name, t] = r.x_disc_con

        result_expr[name, t] = r.result_expr

        for j in 1:depth
            bx_y_u[name, j, t] = r.bx_y.u_var[j]
            bx_y_mc[name, j, 1, t] = r.bx_y.mc_cons[j, 1]
            bx_y_mc[name, j, 2, t] = r.bx_y.mc_cons[j, 2]
            bx_y_mc[name, j, 3, t] = r.bx_y.mc_cons[j, 3]
            bx_y_mc[name, j, 4, t] = r.bx_y.mc_cons[j, 4]
        end
        bx_y_expr[name, t] = r.bx_y.result_expr

        dz_var_c[name, t] = r.dz.z_var
    end

    return result_expr
end
