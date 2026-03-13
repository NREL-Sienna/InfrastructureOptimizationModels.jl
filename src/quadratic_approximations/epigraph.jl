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
struct EpigraphTangentExpression <: ExpressionType end

"""
    _add_epigraph_quadratic_approx!(container, C, names, time_steps, x_var, x_min, x_max, depth, meta)

Create a variable z that lower-bounds x² using tangent-line cuts (Q^{L1} relaxation).

For each (name, t), creates a variable z and adds 2^depth + 1 tangent-line
constraints of the form `z ≥ 2·aₖ·x − aₖ²` at uniformly spaced breakpoints
aₖ = x_min + k·Δ/2^depth for k = 0,…,2^depth. Pure LP — zero binary variables.

Stores affine expressions that lower-bound x² in an `EpigraphExpression` expression container.

The maximum underestimation gap between the tangent envelope and x² is
Δ²·2^{−2·depth−2} where Δ = x_max − x_min.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `depth::Int`: epigraph depth L1 (uses 2^depth + 1 tangent breakpoints)
- `meta::String`: variable type identifier for the approximated variable
"""
function _add_epigraph_quadratic_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op depth >= 1
    jump_model = get_jump_model(container)
    delta = x_max - x_min
    g_levels = 0:depth

    z_var = add_variable_container!(
        container,
        EpigraphVariable(),
        C,
        names,
        time_steps;
        meta
    )
    g_var = add_variable_container!(
        container,
        SawtoothAuxVariable(),
        C,
        names,
        g_levels,
        time_steps;
        meta
    )
    lp_cons = add_constraints_container!(
        container,
        SawtoothLPConstraint(),
        C,
        names,
        1:2,
        time_steps;
        meta
    )
    link_cons = add_constraints_container!(
        container,
        SawtoothLinkingConstraint(),
        C,
        names,
        time_steps;
        meta
    )
    fL_expr = add_expression_container!(
        container,
        EpigraphTangentExpression(),
        C,
        names,
        time_steps;
        meta
    )
    tangent_cons = add_constraints_container!(
        container,
        EpigraphTangentConstraint(),
        C,
        names,
        1:(depth + 2),
        time_steps;
        sparse = true,
        meta
    )
    result_expr = add_expression_container!(
        container,
        EpigraphExpression(),
        C,
        names,
        time_steps;
        meta
    )

    # Upper bound for epigraph variable z ≈ x²
    z_ub = max(x_min^2, x_max^2)

    for name in names, t in time_steps
        x = x_var[name, t]

        # Auxiliary variables g_0,...,g_L ∈ [0, 1]
        for j in g_levels
            g_var[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "SawtoothAux_$(C)_{$(name), $(j), $(t)}",
                lower_bound = 0.0,
                upper_bound = 1.0,
            )
        end
        g0 = g_var[name, 0, t]

        # Linking constraint: g_0 = (x - x_min) / Δ
        link_cons[name, t] = JuMP.@constraint(
            jump_model,
            g0 == (x - x_min) / delta,
        )

        # T^L constraints for j = 1,...,L
        for j in 1:depth
            g_prev = g_var[name, j - 1, t]
            g_curr = g_var[name, j, t]

            # g_j ≤ 2 g_{j-1}
            lp_cons[name, 1, t] = JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev)
            # g_j ≤ 2(1 - g_{j-1})
            lp_cons[name, 2, t] =
                JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev))
        end

        # Create the epigraph variable (bounded from below by tangent cuts)
        z =
            z_var[name, t] = JuMP.@variable(
                jump_model,
                base_name = "EpigraphVar_$(C)_{$(name), $(t)}",
                lower_bound = 0.0,
                upper_bound = z_ub,
            )

        fL = fL_expr[name, t] = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(fL, delta * delta * 2.0^(-2j), g_var[name, j, t])
            tangent_cons[(name, j + 1, t)] = JuMP.@constraint(
                jump_model,
                z >=
                x_min * (2 * delta * g0 + x_min) - fL + delta^2 * (g0 - 2.0^(-2j - 2))
            )
        end
        tangent_cons[name, 1, t] = JuMP.@constraint(jump_model, z >= 0)
        tangent_cons[name, depth + 1, t] = JuMP.@constraint(
            jump_model,
            z >= 2.0 * x_min - 1.0 + 2.0 * delta * g0
        )

        result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
    end

    return result_expr
end
