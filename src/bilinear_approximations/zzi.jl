# ZZI (Integer Zig-Zag) standalone bilinear approximation of x·y.
# Builds a bivariate piecewise linear approximation using a 2D lambda grid with
# ZZI integer-variable SOS2 encoding on each axis and triangle-selection constraints.
# Optionally adds McCormick envelope cuts and sawtooth-based strengthening bounds.
# Reference: Huchette & Vielma, "Nonconvex piecewise linear functions: Optimization
# and polyhedral branching" (arXiv:1708.00050v3, 2019).

# --- Type definitions ---

"Lambda (convex combination weight) variables for the bivariate ZZI grid."
struct ZZILambdaVariable <: SparseVariableType end
"Linking constraint: x == Σ_{i,j} λ_{i,j} * x_bkpts[i]."
struct ZZILinkingXConstraint <: ConstraintType end
"Linking constraint: y == Σ_{i,j} λ_{i,j} * y_bkpts[j]."
struct ZZILinkingYConstraint <: ConstraintType end
"Normalization constraint: Σ_{i,j} λ_{i,j} == 1."
struct ZZINormConstraint <: ConstraintType end
"Product equality constraint: z == Σ_{i,j} λ_{i,j} * x_bkpts[i] * y_bkpts[j]."
struct ZZIProductConstraint <: ConstraintType end

"""
Config for ZZI bilinear approximation.

Uses a (d1+1) × (d2+1) lambda grid with ZZI integer-variable SOS2 encoding on
each axis and a triangle-selection (6-stencil 3-coloring biclique cover).

# Fields
- `d1::Int`: number of intervals along the x-axis
- `d2::Int`: number of intervals along the y-axis
- `add_mccormick::Bool`: whether to add McCormick envelope cuts (default true)
- `add_sawtooth_strengthening::Bool`: whether to add sawtooth-based HybS bounds (default false)
- `sawtooth_depth::Int`: depth for sawtooth quadratic approximation in strengthening
"""
struct ZZIBilinearConfig <: BilinearApproxConfig
    d1::Int
    d2::Int
    add_mccormick::Bool
    add_sawtooth_strengthening::Bool
    sawtooth_depth::Int
end

ZZIBilinearConfig(d1::Int, d2::Int) = ZZIBilinearConfig(d1, d2, true, false, 0)
ZZIBilinearConfig(d1::Int, d2::Int, add_mccormick::Bool) =
    ZZIBilinearConfig(d1, d2, add_mccormick, false, 0)

# --- Main dispatch ---

"""
    _add_bilinear_approx!(config::ZZIBilinearConfig, container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta)

Approximate x·y using the ZZI bivariate piecewise linear formulation.

Builds a (d1+1) × (d2+1) lambda grid, enforces convex-combination linking and
normalization, adds an exact product variable z with equality constraint, and
applies ZZI integer-variable SOS2 encoding plus triangle selection on the grid.

# Arguments
- `config::ZZIBilinearConfig`: grid sizes, McCormick and strengthening options
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `depth::Int`: unused (grid sizes come from config); present for interface compatibility
- `meta::String`: identifier for container keys
"""
function _add_bilinear_approx!(
    config::ZZIBilinearConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op y_max > y_min
    IS.@assert_op config.d1 >= 2
    IS.@assert_op config.d2 >= 2

    d1 = config.d1
    d2 = config.d2

    # Build uniformly spaced breakpoints
    x_bkpts = [x_min + (i - 1) * (x_max - x_min) / d1 for i in 1:(d1 + 1)]
    y_bkpts = [y_min + (j - 1) * (y_max - y_min) / d2 for j in 1:(d2 + 1)]

    # Choose triangulation (d1 × d2 matrix of :U or :K)
    triang = _choose_triangulation(x_bkpts, y_bkpts)

    # Build ZZI encodings for each axis
    r1 = ceil(Int, log2(d1))
    r2 = ceil(Int, log2(d2))
    _, C_ext_x = build_zzi_encoding(d1)
    _, C_ext_y = build_zzi_encoding(d2)

    jump_model = get_jump_model(container)

    # --- Create lambda variable containers ---
    # SparseVariableType container uses linearized grid index (i-1)*(d2+1)+j
    # to fit the 3D (name, idx, t) sparse array format.
    lambda_container = add_variable_container!(container, ZZILambdaVariable(), C; meta)

    # Dict uses (name, i, j, t) tuples for downstream SOS2/triangle functions
    lambda_dict = Dict{Tuple{String, Int, Int, Int}, JuMP.VariableRef}()

    for name in names, t in time_steps
        for i in 1:(d1 + 1), j in 1:(d2 + 1)
            linear_idx = (i - 1) * (d2 + 1) + j
            lam = lambda_container[(name, linear_idx, t)] = JuMP.@variable(
                jump_model,
                base_name = "ZZILambda_$(C)_{$(name), $(i), $(j), $(t)}",
                lower_bound = 0.0,
                upper_bound = 1.0,
            )
            lambda_dict[(name, i, j, t)] = lam
        end
    end

    # --- Linking constraints: x == Σ λ_{i,j} * x_bkpts[i] ---
    link_x_cons = add_constraints_container!(
        container,
        ZZILinkingXConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    link_y_cons = add_constraints_container!(
        container,
        ZZILinkingYConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    norm_cons = add_constraints_container!(
        container,
        ZZINormConstraint(),
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        link_x = JuMP.AffExpr(0.0)
        link_y = JuMP.AffExpr(0.0)
        norm_sum = JuMP.AffExpr(0.0)

        for i in 1:(d1 + 1), j in 1:(d2 + 1)
            lam = lambda_dict[(name, i, j, t)]
            xi = x_bkpts[i]
            yj = y_bkpts[j]
            if xi != 0.0
                add_proportional_to_jump_expression!(link_x, lam, xi)
            end
            if yj != 0.0
                add_proportional_to_jump_expression!(link_y, lam, yj)
            end
            add_proportional_to_jump_expression!(norm_sum, lam, 1.0)
        end

        link_x_cons[name, t] = JuMP.@constraint(jump_model, x_var[name, t] == link_x)
        link_y_cons[name, t] = JuMP.@constraint(jump_model, y_var[name, t] == link_y)
        norm_cons[name, t] = JuMP.@constraint(jump_model, norm_sum == 1.0)
    end

    # --- Create z variable and product equality constraint ---
    z_lo = min(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    z_hi = max(x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)

    z_var = add_variable_container!(
        container,
        BilinearProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    prod_cons = add_constraints_container!(
        container,
        ZZIProductConstraint(),
        C,
        names,
        time_steps;
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

    for name in names, t in time_steps
        z = z_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "ZZI_z_$(C)_{$(name), $(t)}",
            lower_bound = z_lo,
            upper_bound = z_hi,
        )

        # z == Σ_{i,j} λ_{i,j} * x_bkpts[i] * y_bkpts[j]
        prod_sum = JuMP.AffExpr(0.0)
        for i in 1:(d1 + 1), j in 1:(d2 + 1)
            coeff = x_bkpts[i] * y_bkpts[j]
            if coeff != 0.0
                add_proportional_to_jump_expression!(
                    prod_sum, lambda_dict[(name, i, j, t)], coeff,
                )
            end
        end
        prod_cons[name, t] = JuMP.@constraint(jump_model, z == prod_sum)

        result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
    end

    # --- ZZI SOS2 encoding on each axis ---
    _add_zzi_sos2!(
        container, C, names, time_steps,
        lambda_dict, d1, 1, d2 + 1, r1, C_ext_x,
        meta * "_xsos2",
    )
    _add_zzi_sos2!(
        container, C, names, time_steps,
        lambda_dict, d2, 2, d1 + 1, r2, C_ext_y,
        meta * "_ysos2",
    )

    # --- Triangle selection (6-stencil biclique cover) ---
    _add_triangle_selection!(
        container, C, names, time_steps,
        lambda_dict, d1, d2, triang,
        meta * "_tri",
    )

    # --- Optional McCormick envelope ---
    if config.add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, z_var,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    # --- Optional sawtooth strengthening ---
    if config.add_sawtooth_strengthening
        _add_zzi_sawtooth_strengthening!(
            container, C, names, time_steps,
            x_var, y_var, z_var,
            x_min, x_max, y_min, y_max,
            config.sawtooth_depth, meta * "_st",
        )
    end

    return result_expr
end

# --- Sawtooth strengthening ---

"Two-sided HybS-style bound constraints added by ZZI sawtooth strengthening."
struct ZZISawtoothBoundConstraint <: ConstraintType end

"""
    _add_zzi_sawtooth_strengthening!(container, C, names, time_steps, x_var, y_var, z_var, x_min, x_max, y_min, y_max, sawtooth_depth, meta)

Add sawtooth-based HybS strengthening bounds on the ZZI bilinear product variable.

Uses sawtooth (MIP) approximation for x² and y², and epigraph (LP) lower bounds
for (x+y)² and (x−y)², then adds two-sided bounds:
- Lower (Bin2): z ≥ ½(z_{x+y} − z_x − z_y)
- Upper (Bin3): z ≤ ½(z_x + z_y − z_{x−y})

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `z_var`: container of z variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `sawtooth_depth::Int`: number of binary levels for sawtooth MIP approximation of x² and y²
- `meta::String`: base identifier for container keys
"""
function _add_zzi_sawtooth_strengthening!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    z_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    sawtooth_depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op sawtooth_depth >= 1

    p1_min = x_min + y_min
    p1_max = x_max + y_max
    p2_min = x_min - y_max
    p2_max = x_max - y_min

    # MIP sawtooth approximations for x² and y² (tightened with epigraph lower bound
    # to ensure LP relaxation consistency with the ZZI-determined z variable)
    st_config = SawtoothQuadConfig(true, false)
    xsq = _add_quadratic_approx!(
        st_config, container, C, names, time_steps,
        x_var, x_min, x_max, sawtooth_depth, meta * "_x",
    )
    ysq = _add_quadratic_approx!(
        st_config, container, C, names, time_steps,
        y_var, y_min, y_max, sawtooth_depth, meta * "_y",
    )

    # Build p1 = x + y and p2 = x − y expression containers
    p1_expr = add_expression_container!(
        container,
        VariableSumExpression(),
        C,
        names,
        time_steps;
        meta = meta * "_plus",
    )
    p2_expr = add_expression_container!(
        container,
        VariableDifferenceExpression(),
        C,
        names,
        time_steps;
        meta = meta * "_diff",
    )
    for name in names, t in time_steps
        x = x_var[name, t]
        y = y_var[name, t]

        p1 = p1_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p1, x, 1.0)
        add_proportional_to_jump_expression!(p1, y, 1.0)

        p2 = p2_expr[name, t] = JuMP.AffExpr(0.0)
        add_proportional_to_jump_expression!(p2, x, 1.0)
        add_proportional_to_jump_expression!(p2, y, -1.0)
    end

    # LP epigraph lower bounds for (x+y)² and (x−y)²
    ep_config = EpigraphQuadConfig()
    epigraph_depth = max(sawtooth_depth, 1)
    zp1_expr = _add_quadratic_approx!(
        ep_config, container, C, names, time_steps,
        p1_expr, p1_min, p1_max, epigraph_depth, meta * "_plus",
    )
    zp2_expr = _add_quadratic_approx!(
        ep_config, container, C, names, time_steps,
        p2_expr, p2_min, p2_max, epigraph_depth, meta * "_diff",
    )

    # Two-sided HybS-style bound constraints
    st_cons = add_constraints_container!(
        container,
        ZZISawtoothBoundConstraint(),
        C,
        names,
        1:2,
        time_steps;
        sparse = true,
        meta,
    )

    jump_model = get_jump_model(container)
    for name in names, t in time_steps
        z = z_var[name, t]
        zx = xsq[name, t]
        zy = ysq[name, t]
        zp1 = zp1_expr[name, t]
        zp2 = zp2_expr[name, t]

        # Bin2 lower bound: z >= ½(z_{x+y} − z_x − z_y)
        st_cons[(name, 1, t)] = JuMP.@constraint(
            jump_model,
            z >= 0.5 * (zp1 - zx - zy),
        )
        # Bin3 upper bound: z <= ½(z_x + z_y − z_{x−y})
        st_cons[(name, 2, t)] = JuMP.@constraint(
            jump_model,
            z <= 0.5 * (zx + zy - zp2),
        )
    end

    return
end
