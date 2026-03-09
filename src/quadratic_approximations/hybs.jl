# HybS (Hybrid Separable) MIP relaxation for bilinear products z = x·y.
# Combines Bin2 lower bound and Bin3 upper bound with shared sawtooth for x², y²
# and LP-only epigraph for (x+y)², (x−y)². Uses 2L binaries instead of 3L (Bin2).
# Reference: Beach, Burlacu, Bärmann, Hager, Hildebrand (2024), Definition 10.

"Expression container for HybS bilinear product approximation results."
struct HybSProductExpression <: ExpressionType end

"Auxiliary variable p₂ = x − y for HybS bilinear approximation."
struct HybSApproxDiffVariable <: VariableType end
"Links p₂ = x − y in HybS bilinear approximation."
struct HybSApproxDiffLinkingConstraint <: ConstraintType end
"Two-sided HybS bound constraints: Bin2 lower + Bin3 upper."
struct HybSBoundConstraint <: ConstraintType end

"""
    _add_hybs_bilinear_approx!(container, C, names, time_steps, x_var_container, y_var_container, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate x·y using the HybS (Hybrid Separable) relaxation from Beach et al. (2024).

Combines Bin2 and Bin3 separable identities:
- Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y) where z_p1 lower-bounds (x+y)²
- Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2) where z_p2 lower-bounds (x−y)²

Only x² and y² use the full sawtooth S^L (with L binary variables each).
The cross-terms (x+y)² and (x−y)² use only epigraph Q^{L1} (pure LP).

Stores affine expressions approximating x·y in a `HybSProductExpression` expression container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var_container`: container of x variables indexed by (name, t)
- `y_var_container`: container of y variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `depth::Int`: sawtooth depth L (number of binary variables per x²/y² approximation)
- `meta::String`: identifier encoding the original variable type being approximated
- `add_mccormick::Bool`: whether to add McCormick envelope constraints (default: true)
"""
function _add_hybs_bilinear_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    y_var_container,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    add_mccormick::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    # Bounds for auxiliary variables
    p1_min = x_min + y_min
    p1_max = x_max + y_max
    p2_min = x_min - y_max     # p2 = x − y, so min uses −y_max
    p2_max = x_max - y_min     # and max uses −y_min
    IS.@assert_op x_max > x_min
    IS.@assert_op y_max > y_min

    jump_model = get_jump_model(container)

    # Meta suffixes: key = (ApproxType, ComponentType, OriginalVarType_suffix)
    meta_x = meta * "_x"
    meta_y = meta * "_y"
    meta_p1 = meta * "_plus"
    meta_p2 = meta * "_diff"

    # --- Step 1: Auxiliary sum variable p1 = x + y ---
    p1_container = add_variable_container!(
        container,
        BilinearApproxSumVariable(),
        C,
        names,
        time_steps;
        meta = meta_p1,
    )
    p1_link_container = add_constraints_container!(
        container,
        BilinearApproxSumLinkingConstraint(),
        C,
        names,
        time_steps;
        meta = meta_p1,
    )

    # --- Step 2: Auxiliary difference variable p2 = x − y ---
    p2_container = add_variable_container!(
        container,
        HybSApproxDiffVariable(),
        C,
        names,
        time_steps;
        meta = meta_p2,
    )
    p2_link_container = add_constraints_container!(
        container,
        HybSApproxDiffLinkingConstraint(),
        C,
        names,
        time_steps;
        meta = meta_p2,
    )

    for name in names, t in time_steps
        x = x_var_container[name, t]
        y = y_var_container[name, t]

        # p1 = x + y
        p1_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "HybSSum_$(C)_{$(name), $(t)}",
            lower_bound = p1_min,
            upper_bound = p1_max,
        )
        p1_link_container[name, t] =
            JuMP.@constraint(jump_model, p1_container[name, t] == x + y)

        # p2 = x − y
        p2_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "HybSDiff_$(C)_{$(name), $(t)}",
            lower_bound = p2_min,
            upper_bound = p2_max,
        )
        p2_link_container[name, t] =
            JuMP.@constraint(jump_model, p2_container[name, t] == x - y)
    end

    # --- Step 3: Sawtooth S^L upper bound for x² and y² (binary variables here) ---
    _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        x_var_container, x_min, x_max, depth, meta_x,
    )
    _add_sawtooth_quadratic_approx!(
        container, C, names, time_steps,
        y_var_container, y_min, y_max, depth, meta_y,
    )

    # --- Step 4: Epigraph Q^{L1} lower bound for (x+y)² and (x−y)² (no binaries) ---
    _add_epigraph_quadratic_approx!(
        container, C, names, time_steps,
        p1_container, p1_min, p1_max, depth, meta_p1,
    )
    _add_epigraph_quadratic_approx!(
        container, C, names, time_steps,
        p2_container, p2_min, p2_max, depth, meta_p2,
    )

    # Retrieve expression containers
    zx_container = get_expression(
        container, QuadraticApproximationExpression(), C, meta_x,
    )
    zy_container = get_expression(
        container, QuadraticApproximationExpression(), C, meta_y,
    )
    zp1_container = get_expression(
        container, EpigraphExpression(), C, meta_p1,
    )
    zp2_container = get_expression(
        container, EpigraphExpression(), C, meta_p2,
    )

    # --- Step 5: Create z variable and two-sided HybS bounds ---
    z_container = add_variable_container!(
        container,
        BilinearProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )
    hybs_bound_container = add_constraints_container!(
        container,
        HybSBoundConstraint(),
        C,
        names,
        1:2,
        time_steps;
        meta,
        sparse = true,
    )

    expr_container = add_expression_container!(
        container,
        HybSProductExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        z_var = JuMP.@variable(
            jump_model,
            base_name = "HybSProduct_$(C)_{$(name), $(t)}",
        )
        z_container[name, t] = z_var

        zx = zx_container[name, t]
        zy = zy_container[name, t]
        zp1 = zp1_container[name, t]
        zp2 = zp2_container[name, t]

        # Bin2 lower bound: z ≥ ½(z_p1 − z_x − z_y)
        hybs_bound_container[(name, 1, t)] = JuMP.@constraint(
            jump_model,
            z_var >= 0.5 * (zp1 - zx - zy),
        )
        # Bin3 upper bound: z ≤ ½(z_x + z_y − z_p2)
        hybs_bound_container[(name, 2, t)] = JuMP.@constraint(
            jump_model,
            z_var <= 0.5 * (zx + zy - zp2),
        )

        expr_container[name, t] = JuMP.AffExpr(0.0, z_var => 1.0)
    end

    # --- Step 6: McCormick envelope for additional tightening ---
    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var_container, y_var_container, z_container,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    return
end
