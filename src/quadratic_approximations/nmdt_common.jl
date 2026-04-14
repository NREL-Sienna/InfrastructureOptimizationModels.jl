"Binary discretization variables β_i ∈ {0,1} in the NMDT decomposition of xh."
struct NMDTBinaryVariable <: VariableType end
"Residual variable δ ∈ [0, 2^{−L}] capturing the NMDT discretization error."
struct NMDTResidualVariable <: VariableType end
"McCormick linearization variables u_i ≈ β_i · y in NMDT binary-continuous products."
struct NMDTBinaryContinuousProductVariable <: VariableType end
"Variable z ≈ δ · y linearizing the residual-continuous product in NMDT."
struct NMDTResidualProductVariable <: VariableType end

"Expression container for the NMDT binary discretization: Σ 2^{−i}·β_i + δ ≈ xh."
struct NMDTDiscretizationExpression <: ExpressionType end
"Expression container for the NMDT binary-continuous product: Σ 2^{−i}·u_i ≈ β·y."
struct NMDTBinaryContinuousProductExpression <: ExpressionType end
"Expression container for the final NMDT quadratic approximation result."
struct NMDTResultExpression <: ExpressionType end

"Constraint enforcing xh = Σ 2^{−i}·β_i + δ in the NMDT discretization."
struct NMDTEDiscretizationConstraint <: ConstraintType end
"McCormick envelope constraints for binary-continuous products u_i ≈ β_i·y in NMDT."
struct NMDTBinaryContinuousProductConstraint <: ConstraintType end
"Epigraph lower-bound tightening constraint on the NMDT quadratic result."
struct NMDTTightenConstraint <: ConstraintType end

"""
Stores the per-(name,t) result of discretizing a normalized variable for NMDT products.

Fields:
- `norm_expr::JuMP.AffExpr`: affine expression xh = (x − x_min)/(x_max − x_min) ∈ [0,1]
- `beta_var::Vector{JuMP.VariableRef}`: binary variables β₁,…,β_L
- `delta_var::JuMP.VariableRef`: residual variable δ ∈ [0, 2^{−depth}]
"""
struct NMDTDiscretization
    norm_expr::JuMP.AffExpr
    beta_var::Vector{JuMP.VariableRef}
    delta_var::JuMP.VariableRef
end

"""
    _discretize!(jump_model, x, bounds, depth, meta)

Discretize the normalized variable xh = (x − x_min)/(x_max − x_min) using L binary variables.

Creates L binary variables β₁,…,β_L and one residual δ ∈ [0, 2^{−L}] such that
xh = Σᵢ 2^{−i}·β_i + δ. Returns an `NMDTDiscretization` holding all components,
plus the discretization expression and enforcing constraint.

# Arguments
- `jump_model::JuMP.Model`: the JuMP optimization model
- `x::JuMP.AbstractJuMPScalar`: single variable to discretize
- `bounds::MinMax`: `(min = x_min, max = x_max)` bounds for x
- `depth::Int`: number of binary discretization levels L
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple `(disc, disc_expr, disc_con)` where:
- `disc::NMDTDiscretization`: struct with `norm_expr`, `beta_var`, `delta_var`
- `disc_expr::JuMP.AffExpr`: the discretization expression Σ 2^{−i}·β_i + δ
- `disc_con::JuMP.ConstraintRef`: constraint enforcing xh == disc_expr
"""
function _discretize!(
    jump_model::JuMP.Model,
    x::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    depth::Int,
    meta::String,
)
    IS.@assert_op bounds.max > bounds.min
    IS.@assert_op depth >= 1

    xh = _normed_variable(x, bounds)

    betas = Vector{JuMP.VariableRef}(undef, depth)
    disc = JuMP.AffExpr(0.0)
    for i in 1:depth
        beta = JuMP.@variable(
            jump_model,
            base_name = "NMDTBin_$(meta)_$(i)",
            binary = true
        )
        betas[i] = beta
        add_proportional_to_jump_expression!(disc, beta, 2.0^(-i))
    end
    delta = JuMP.@variable(
        jump_model,
        base_name = "NMDTRes_$(meta)",
        lower_bound = 0.0,
        upper_bound = 2.0^(-depth)
    )
    add_proportional_to_jump_expression!(disc, delta, 1.0)
    con = JuMP.@constraint(jump_model, xh == disc)

    return (
        disc = NMDTDiscretization(xh, betas, delta),
        disc_expr = disc,
        disc_con = con,
    )
end

"""
    _binary_continuous_product!(jump_model, disc, cont_var, cont_min, cont_max, depth, meta; tighten)

Linearize each binary-continuous product β_i·y using McCormick envelopes for a single (name,t).

For each depth level i, creates a variable u_i ≈ β_i·y with bounds [cont_min, cont_max]
and adds 4 McCormick constraints. Assembles the weighted sum Σᵢ 2^{−i}·u_i.

# Arguments
- `jump_model::JuMP.Model`: the JuMP optimization model
- `disc::NMDTDiscretization`: per-(name,t) discretization providing β_i variables
- `cont_var::JuMP.AbstractJuMPScalar`: single continuous variable y
- `cont_min::Float64`: lower bound of y
- `cont_max::Float64`: upper bound of y
- `depth::Int`: number of binary discretization levels
- `meta::String`: base name prefix for JuMP variables
- `tighten::Bool`: if true, omit McCormick lower bounds (default: false)

# Returns
Named tuple `(u_var, mc_cons, result_expr)` where:
- `u_var::Vector{JuMP.VariableRef}`: McCormick auxiliary variables u_1,…,u_L
- `mc_cons::Matrix{JuMP.ConstraintRef}`: McCormick constraints (depth × 4)
- `result_expr::JuMP.AffExpr`: weighted sum Σ 2^{−i}·u_i
"""
function _binary_continuous_product!(
    jump_model::JuMP.Model,
    disc::NMDTDiscretization,
    cont_var::JuMP.AbstractJuMPScalar,
    cont_min::Float64,
    cont_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
)
    u_vars = Vector{JuMP.VariableRef}(undef, depth)
    mc_cons = Matrix{JuMP.ConstraintRef}(undef, depth, 4)
    result = JuMP.AffExpr(0.0)

    for i in 1:depth
        u_i = JuMP.@variable(
            jump_model,
            base_name = "NMDTBinCont_$(meta)_$(i)",
            lower_bound = cont_min,
            upper_bound = cont_max
        )
        u_vars[i] = u_i
        beta_i = disc.beta_var[i]

        # McCormick envelope: u_i ≈ cont_var * beta_i
        # x = cont_var ∈ [cont_min, cont_max], y = beta_i ∈ [0, 1]
        if !tighten
            mc_cons[i, 1] = JuMP.@constraint(
                jump_model,
                u_i >= cont_min * beta_i,
            )
            mc_cons[i, 2] = JuMP.@constraint(
                jump_model,
                u_i >= cont_max * beta_i + cont_var - cont_max,
            )
        end
        mc_cons[i, 3] = JuMP.@constraint(
            jump_model,
            u_i <= cont_max * beta_i,
        )
        mc_cons[i, 4] = JuMP.@constraint(
            jump_model,
            u_i <= cont_min * beta_i + cont_var - cont_min,
        )

        add_proportional_to_jump_expression!(result, u_i, 2.0^(-i))
    end

    return (u_var = u_vars, mc_cons = mc_cons, result_expr = result)
end

"""
    _residual_product!(jump_model, disc, y, y_max, depth, meta; tighten)

Linearize the residual-continuous product z ≈ δ·y using McCormick envelopes for a single (name,t).

Creates a variable z ∈ [0, 2^{−L}·y_max] and bounds it with McCormick constraints
on (δ, y) where δ = disc.delta_var ∈ [0, 2^{−L}].

# Arguments
- `jump_model::JuMP.Model`: the JuMP optimization model
- `disc::NMDTDiscretization`: per-(name,t) discretization providing δ variable
- `y::JuMP.AbstractJuMPScalar`: continuous variable y
- `y_max::Float64`: upper bound of y (lower bound assumed 0)
- `depth::Int`: number of binary discretization levels
- `meta::String`: base name prefix for JuMP variables
- `tighten::Bool`: if true, omit McCormick lower bounds (default: false)

# Returns
Named tuple `(z_var, mc_cons)` where:
- `z_var::JuMP.VariableRef`: residual product variable z
- `mc_cons::Vector{JuMP.ConstraintRef}`: McCormick constraints (length 4)
"""
function _residual_product!(
    jump_model::JuMP.Model,
    disc::NMDTDiscretization,
    y::JuMP.AbstractJuMPScalar,
    y_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
)
    x_max = 2.0^(-depth)
    delta = disc.delta_var

    z = JuMP.@variable(
        jump_model,
        base_name = "NMDTResProd_$(meta)",
        lower_bound = 0.0,
        upper_bound = x_max * y_max,
    )

    # McCormick envelope: z ≈ delta * y
    # delta ∈ [0, x_max], y ∈ [0, y_max]
    mc = Vector{JuMP.ConstraintRef}(undef, 4)
    if !tighten
        # z >= x_min*y + x*y_min - x_min*y_min  =>  z >= 0 (trivial, but kept for completeness)
        mc[1] = JuMP.@constraint(jump_model, z >= 0.0)
        # z >= x_max*y + x*y_max - x_max*y_max  =>  z >= x_max*y + y_max*delta - x_max*y_max
        mc[2] = JuMP.@constraint(
            jump_model,
            z >= x_max * y + y_max * delta - x_max * y_max,
        )
    end
    # z <= x_max*y + x*y_min - x_max*y_min  =>  z <= x_max*y (since y_min = 0)
    mc[3] = JuMP.@constraint(jump_model, z <= x_max * y)
    # z <= x_min*y + x*y_max - x_min*y_max  =>  z <= y_max*delta (since x_min = 0)
    mc[4] = JuMP.@constraint(jump_model, z <= y_max * delta)

    return (z_var = z, mc_cons = mc)
end

"""
    _assemble_product(terms, dz, xh, yh, x_bounds, y_bounds)

Reconstruct the bilinear product x·y from normalized NMDT components (pure computation).

Applies the affine rescaling:
```
x·y = lx·ly·zh + lx·y_min·xh + ly·x_min·yh + x_min·y_min
```
where `zh = Σ terms + dz` collects the binary-continuous and residual product contributions,
lx = x_max − x_min, ly = y_max − y_min.

# Arguments
- `terms`: iterable of `JuMP.AffExpr` values for the binary-continuous products
- `dz::JuMP.AbstractJuMPScalar`: residual product contribution δ·y
- `xh::JuMP.AffExpr`: normalized expression for x
- `yh::JuMP.AffExpr`: normalized expression for y
- `x_bounds::MinMax`: `(min, max)` bounds for x
- `y_bounds::MinMax`: `(min, max)` bounds for y

# Returns
`JuMP.AffExpr` for the reconstructed product x·y.
"""
function _assemble_product(
    terms,
    dz::JuMP.AbstractJuMPScalar,
    xh::JuMP.AffExpr,
    yh::JuMP.AffExpr,
    x_bounds::MinMax,
    y_bounds::MinMax,
)
    lx = x_bounds.max - x_bounds.min
    ly = y_bounds.max - y_bounds.min

    zh = JuMP.AffExpr(0.0)
    for term in terms
        add_proportional_to_jump_expression!(zh, term, 1.0)
    end
    add_proportional_to_jump_expression!(zh, dz, 1.0)

    result = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(result, zh, lx * ly)
    add_proportional_to_jump_expression!(result, xh, lx * y_bounds.min)
    add_proportional_to_jump_expression!(result, yh, ly * x_bounds.min)
    add_constant_to_jump_expression!(result, x_bounds.min * y_bounds.min)

    return result
end

"""
    _assemble_dnmdt!(jump_model, bx_yh, by_dx, by_xh, bx_dy, x_disc, y_disc, x_bounds, y_bounds, depth, meta; lambda, tighten)

Core assembler for the DNMDT bilinear approximation of x·y from pre-computed cross products
for a single (name,t).

Builds two NMDT product estimates from opposite discretization pairings and combines them:
- z₁ = assemble(bx·yh + by·δx, residual δx·δy, x_disc, y_disc)
- z₂ = assemble(by·xh + bx·δy, residual δx·δy, y_disc, x_disc)
- result = λ·z₁ + (1−λ)·z₂

# Arguments
- `jump_model::JuMP.Model`: the JuMP optimization model
- `bx_yh::JuMP.AffExpr`: expression for β_x·yh binary-continuous product
- `by_dx::JuMP.AffExpr`: expression for β_y·δx binary-continuous product
- `by_xh::JuMP.AffExpr`: expression for β_y·xh binary-continuous product
- `bx_dy::JuMP.AffExpr`: expression for β_x·δy binary-continuous product
- `x_disc::NMDTDiscretization`: per-(name,t) discretization for x
- `y_disc::NMDTDiscretization`: per-(name,t) discretization for y
- `x_bounds::MinMax`: `(min, max)` bounds for x
- `y_bounds::MinMax`: `(min, max)` bounds for y
- `depth::Int`: number of binary discretization levels
- `meta::String`: base name prefix for JuMP variables
- `lambda::Float64`: convex combination weight (default: `DNMDT_LAMBDA`)
- `tighten::Bool`: if true, omit McCormick lower bounds in residual product (default: false)

# Returns
Named tuple `(result_expr, dz_result, z1_expr, z2_expr)` where:
- `result_expr::JuMP.AffExpr`: final DNMDT approximation λ·z₁ + (1−λ)·z₂
- `dz_result`: named tuple from `_residual_product!` (z_var, mc_cons)
- `z1_expr::JuMP.AffExpr`: first NMDT product estimate
- `z2_expr::JuMP.AffExpr`: second NMDT product estimate
"""
function _assemble_dnmdt!(
    jump_model::JuMP.Model,
    bx_yh::JuMP.AffExpr,
    by_dx::JuMP.AffExpr,
    by_xh::JuMP.AffExpr,
    bx_dy::JuMP.AffExpr,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    x_bounds::MinMax,
    y_bounds::MinMax,
    depth::Int,
    meta::String;
    lambda::Float64 = DNMDT_LAMBDA,
    tighten::Bool = false,
)
    dz_result = _residual_product!(
        jump_model, x_disc,
        y_disc.delta_var, 2.0^(-depth),
        depth, meta; tighten,
    )

    z1 = _assemble_product(
        [bx_yh, by_dx], dz_result.z_var,
        x_disc.norm_expr, y_disc.norm_expr,
        x_bounds, y_bounds,
    )
    z2 = _assemble_product(
        [by_xh, bx_dy], dz_result.z_var,
        y_disc.norm_expr, x_disc.norm_expr,
        y_bounds, x_bounds,
    )

    result = JuMP.AffExpr(0.0)
    add_proportional_to_jump_expression!(result, z1, lambda)
    add_proportional_to_jump_expression!(result, z2, 1.0 - lambda)

    return (result_expr = result, dz_result = dz_result, z1_expr = z1, z2_expr = z2)
end

"""
    _tighten_lower_bounds!(jump_model, result_expr, xh, epigraph_depth, meta)

Add an epigraph lower-bound constraint to tighten an NMDT quadratic approximation
for a single (name,t).

Computes an epigraph Q^{L1} lower bound on xh² and constrains `result_expr ≥ epi_result`.

# Arguments
- `jump_model::JuMP.Model`: the JuMP optimization model
- `result_expr::JuMP.AffExpr`: the NMDT quadratic result to tighten
- `xh::JuMP.AffExpr`: normalized variable expression ∈ [0,1]
- `epigraph_depth::Int`: depth for the epigraph lower-bound approximation
- `meta::String`: base name prefix for JuMP variables

# Returns
Named tuple `(tight_con, epi_result)` where:
- `tight_con::JuMP.ConstraintRef`: constraint enforcing result_expr ≥ epi lower bound
- `epi_result`: result from `_add_quadratic_approx!(EpigraphQuadConfig(...), ...)`
"""
function _tighten_lower_bounds!(
    jump_model::JuMP.Model,
    result_expr::JuMP.AffExpr,
    xh::JuMP.AffExpr,
    epigraph_depth::Int,
    meta::String,
)
    epi_result = _add_quadratic_approx!(
        EpigraphQuadConfig(epigraph_depth),
        jump_model, xh, (min = 0.0, max = 1.0),
        meta * "_epi",
    )

    tight_con = JuMP.@constraint(
        jump_model,
        result_expr >= epi_result.result_expr,
    )

    return (tight_con = tight_con, epi_result = epi_result)
end
