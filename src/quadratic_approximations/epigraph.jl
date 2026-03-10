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

"""
    _add_epigraph_quadratic_approx!(container, C, names, time_steps, x_var_container, x_min, x_max, depth, meta)

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
- `x_var_container`: container of variables indexed by (name, t)
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
    x_var_container,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op depth >= 1
    jump_model = get_jump_model(container)
    delta = x_max - x_min

    # Create z variable container
    z_container = add_variable_container!(
        container,
        EpigraphVariable(),
        C,
        names,
        time_steps;
        meta,
    )

    g_levels = 0:depth
    g_container = add_variable_container!(
        container,
        SawtoothAuxVariable(),
        C,
        names,
        g_levels,
        time_steps;
        meta,
    )
    lp_container = add_constraints_container!(
        container,
        SawtoothLPConstraint(),
        C,
        names,
        1:2,
        time_steps;
        meta,
    )
    link_container = add_constraints_container!(
        container,
        SawtoothLinkingConstraint(),
        C,
        names,
        time_steps;
        meta,
    )

    # Create tangent constraint container (sparse: name × breakpoint_idx × t)
    tangent_container = add_constraints_container!(
        container,
        EpigraphTangentConstraint(),
        C,
        names,
        1:(depth + 2),
        time_steps;
        meta,
        sparse = true,
    )

    expr_container = add_expression_container!(
        container,
        EpigraphExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    # Precompute breakpoint values and upper bound (invariant across names and time steps)
    step = delta / (n_breakpoints - 1)
    breakpoints = [(x_min + (k - 1) * step) for k in 1:n_breakpoints]
    z_ub = max(x_min^2, x_max^2)

    for name in names, t in time_steps
        x_var = x_var_container[name, t]

        # Auxiliary variables g_0,...,g_L ∈ [0, 1]
        for j in g_levels
            g_container[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "SawtoothAux_$(C)_{$(name), $(j), $(t)}",
                lower_bound = 0.0,
                upper_bound = 1.0,
            )
        end
        g0 = g_container[name, 0, t]

        # Linking constraint: g_0 = (x - x_min) / Δ
        link_container[name, t] = JuMP.@constraint(
            jump_model,
            g0 == (x_var - x_min) / delta,
        )

        # T^L constraints for j = 1,...,L
        for j in 1:depth
            g_prev = g_container[name, j - 1, t]
            g_curr = g_container[name, j, t]

            # g_j ≤ 2 g_{j-1}
            lp_container[name, 1, t] = JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev)
            # g_j ≤ 2(1 - g_{j-1})
            lp_container[name, 2, t] =
                JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev))
        end

        # Create the epigraph variable (bounded from below by tangent cuts)
        z_var = JuMP.@variable(
            jump_model,
            base_name = "EpigraphVar_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = z_ub,
        )
        z_container[name, t] = z_var

        fL = JuMP.AffExpr(0.0)
        for j in 1:depth
            JuMP.add_to_expression!(fL, delta * delta * 2.0^(-2j), g_container[name, j, t])
            tangent_container[(name, j + 1, t)] = JuMP.@constraint(
                jump_model,
                z_var >=
                x_min * (2 * delta * g0 + x_min) - fL + delta^2 * (g0 - 2.0^(-2j - 2))
            )
        end
        tangent_container[name, 1, t] = JuMP.@constraint(jump_model, z_var >= 0)
        tangent_container[name, depth + 1, t] = JuMP.@constraint(
            jump_model,
            z_var >= 2.0 * x_min - 1.0 + 2.0 * delta * g_container[name, 0, t]
        )

        expr_container[name, t] = JuMP.AffExpr(0.0, z_var => 1.0)
    end

    return
end
