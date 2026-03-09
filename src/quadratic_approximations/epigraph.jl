# Epigraph (Q^{L1}) LP-only lower bound for x² using tangent-line cuts.
# Pure LP — zero binary variables. Creates a variable z ≥ x² (approximately)
# bounded from below by supporting hyperplanes of the parabola.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024), Q^{L1} relaxation.

"Expression container for epigraph quadratic approximation results."
struct EpigraphExpression <: ExpressionType end

"Variable representing a lower-bounded approximation of x² in epigraph relaxation."
struct EpigraphVariable <: VariableType end
"Tangent-line lower-bound constraints z ≥ 2·a·x − a² in epigraph relaxation."
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
    n_breakpoints = (1 << depth) + 1  # 2^depth + 1

    # Create z variable container
    z_container = add_variable_container!(
        container,
        EpigraphVariable(),
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
        1:n_breakpoints,
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

    # Precompute breakpoint values (invariant across names and time steps)
    step = delta / (n_breakpoints - 1)
    breakpoints = [(x_min + (k - 1) * step) for k in 1:n_breakpoints]

    for name in names, t in time_steps
        x_var = x_var_container[name, t]

        # Create the epigraph variable (bounded from below by tangent cuts)
        z_var = JuMP.@variable(
            jump_model,
            base_name = "EpigraphVar_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
        )
        z_container[name, t] = z_var

        # Add tangent-line lower-bound constraints:
        # z ≥ 2·aₖ·x − aₖ² for each breakpoint aₖ
        for k in 1:n_breakpoints
            a_k = breakpoints[k]
            tangent_container[(name, k, t)] = JuMP.@constraint(
                jump_model,
                z_var >= 2.0 * a_k * x_var - a_k * a_k,
            )
        end

        expr_container[name, t] = JuMP.AffExpr(0.0, z_var => 1.0)
    end

    return
end
