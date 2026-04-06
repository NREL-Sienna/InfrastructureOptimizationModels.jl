# ZZI strengthening wrapper for bilinear approximation.
# Composes any existing BilinearApproxConfig with a redundant ZZI lambda grid as valid
# inequalities to strengthen the LP relaxation and improve branching structure.
# The ZZI constraints are redundant at integer-feasible points but tighten the LP.
# Reference: Huchette & Vielma, "Nonconvex piecewise linear functions: Optimization
# and polyhedral branching" (arXiv:1708.00050v3, 2019).

# --- Type definitions ---

"Redundant product equality linking the inner config's z expression to the ZZI lambda grid."
struct ZZIRedundantProductConstraint <: ConstraintType end

"""
Config that wraps any `BilinearApproxConfig` with a redundant ZZI grid.

Runs the inner config's approximation first, then adds a ZZI lambda grid with SOS2
encoding and triangle selection linked to the same x, y, and z variables. The ZZI
constraints are redundant valid inequalities that strengthen the LP relaxation.

# Fields
- `inner::BilinearApproxConfig`: the primary bilinear approximation method
- `d1::Int`: number of intervals along the x-axis for the ZZI grid
- `d2::Int`: number of intervals along the y-axis for the ZZI grid
"""
struct ZZIStrengthenedConfig <: BilinearApproxConfig
    inner::BilinearApproxConfig
    d1::Int
    d2::Int
end

# --- Main dispatch ---

"""
    _add_bilinear_approx!(config::ZZIStrengthenedConfig, container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta)

Approximate x·y using the inner config, then add a redundant ZZI grid for LP strengthening.

Runs the inner config's `_add_bilinear_approx!` to produce the primary z approximation,
then builds a separate (d1+1) × (d2+1) ZZI lambda grid linked to the same x, y, and z.
The linking constraints (x-link, y-link, norm, product) are added as redundant valid
inequalities together with ZZI SOS2 encoding and triangle selection.

# Arguments
- `config::ZZIStrengthenedConfig`: inner config plus ZZI grid dimensions
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
- `depth::Int`: passed through to the inner config
- `meta::String`: identifier for container keys (inner config uses `meta`, ZZI uses `meta * "_zzir"`)
"""
function _add_bilinear_approx!(
    config::ZZIStrengthenedConfig,
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

    # 1. Run the inner config to produce the primary approximation
    result_expr = _add_bilinear_approx!(
        config.inner,
        container,
        C,
        names,
        time_steps,
        x_var,
        y_var,
        x_min,
        x_max,
        y_min,
        y_max,
        meta,
    )

    d1 = config.d1
    d2 = config.d2

    # Use a distinct meta suffix to avoid container key conflicts with the inner config
    meta_zzi = meta * "_zzir"

    # 2. Build uniformly spaced breakpoints
    x_bkpts = [x_min + (i - 1) * (x_max - x_min) / d1 for i in 1:(d1 + 1)]
    y_bkpts = [y_min + (j - 1) * (y_max - y_min) / d2 for j in 1:(d2 + 1)]

    # 3. Choose triangulation
    triang = _choose_triangulation(x_bkpts, y_bkpts)

    # 4. Build ZZI encodings for each axis
    r1 = ceil(Int, log2(d1))
    r2 = ceil(Int, log2(d2))
    _, C_ext_x = build_zzi_encoding(d1)
    _, C_ext_y = build_zzi_encoding(d2)

    jump_model = get_jump_model(container)

    # 5. Create lambda variable containers (use meta_zzi to avoid conflicts)
    # Linearized grid index (i-1)*(d2+1)+j for the 3D sparse container
    lambda_container = add_variable_container!(container, ZZILambdaVariable(), C; meta = meta_zzi)

    lambda_dict = Dict{Tuple{String, Int, Int, Int}, JuMP.VariableRef}()

    for name in names, t in time_steps
        for i in 1:(d1 + 1), j in 1:(d2 + 1)
            linear_idx = (i - 1) * (d2 + 1) + j
            lam = lambda_container[(name, linear_idx, t)] = JuMP.@variable(
                jump_model,
                base_name = "ZZIrLambda_$(C)_{$(name), $(i), $(j), $(t)}",
                lower_bound = 0.0,
                upper_bound = 1.0,
            )
            lambda_dict[(name, i, j, t)] = lam
        end
    end

    # 6. Add linking constraints for x, y, and normalization
    link_x_cons = add_constraints_container!(
        container,
        ZZILinkingXConstraint(),
        C,
        names,
        time_steps;
        meta = meta_zzi,
    )
    link_y_cons = add_constraints_container!(
        container,
        ZZILinkingYConstraint(),
        C,
        names,
        time_steps;
        meta = meta_zzi,
    )
    norm_cons = add_constraints_container!(
        container,
        ZZINormConstraint(),
        C,
        names,
        time_steps;
        meta = meta_zzi,
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

    # 7. Add redundant product constraint: z_expr == Σ_{i,j} λ_{i,j} * x_i * y_j
    #    z_expr = result_expr[name, t] may be a VariableRef or AffExpr — both work as
    #    AbstractJuMPScalar on the lhs of a JuMP equality constraint.
    prod_cons = add_constraints_container!(
        container,
        ZZIRedundantProductConstraint(),
        C,
        names,
        time_steps;
        meta = meta_zzi,
    )

    for name in names, t in time_steps
        prod_sum = JuMP.AffExpr(0.0)
        for i in 1:(d1 + 1), j in 1:(d2 + 1)
            coeff = x_bkpts[i] * y_bkpts[j]
            if coeff != 0.0
                add_proportional_to_jump_expression!(
                    prod_sum, lambda_dict[(name, i, j, t)], coeff,
                )
            end
        end
        z_expr = result_expr[name, t]
        prod_cons[name, t] = JuMP.@constraint(jump_model, z_expr == prod_sum)
    end

    # 8. ZZI SOS2 encoding on each axis
    _add_zzi_sos2!(
        container, C, names, time_steps,
        lambda_dict, d1, 1, d2 + 1, r1, C_ext_x,
        meta_zzi * "_xsos2",
    )
    _add_zzi_sos2!(
        container, C, names, time_steps,
        lambda_dict, d2, 2, d1 + 1, r2, C_ext_y,
        meta_zzi * "_ysos2",
    )

    # 9. Triangle selection (6-stencil biclique cover)
    _add_triangle_selection!(
        container, C, names, time_steps,
        lambda_dict, d1, d2, triang,
        meta_zzi * "_tri",
    )

    # 10. Return the inner config's result expression (not a new one)
    return result_expr
end
