# Sawtooth MIP approximation of x² for use in constraints.
# Uses recursive tooth function compositions with O(log(1/ε)) binary variables.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024).

struct SawtoothAuxVariable <: VariableType end
struct SawtoothBinaryVariable <: VariableType end
struct SawtoothLinkingConstraint <: ConstraintType end

"""
    _add_sawtooth_quadratic_approx!(container, C, names, time_steps, x_var_container, x_min, x_max, depth, meta)

Approximate x² using the sawtooth MIP formulation.

Creates auxiliary continuous variables g_0,...,g_L and binary variables α_1,...,α_L,
adds S^L constraints (4 per level) and a linking constraint for each component and
time step, and returns a dictionary of JuMP affine expressions approximating x².

For depth L, the approximation interpolates x² at 2^L + 1 uniformly spaced breakpoints
with maximum overestimation error Δ² · 2^{-2L-2} where Δ = x_max - x_min.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var_container`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `depth::Int`: sawtooth depth L (number of binary variables per component per time step)
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)

# Returns
- `Dict{Tuple{String, Int}, JuMP.AffExpr}`: maps (name, t) to affine expression approximating x²
"""
function _add_sawtooth_quadratic_approx!(
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

    # Create containers with known dimensions
    g_levels = 0:depth
    alpha_levels = 1:depth
    g_container = add_variable_container!(
        container,
        SawtoothAuxVariable(),
        C,
        names,
        g_levels,
        time_steps;
        meta,
    )
    alpha_container = add_variable_container!(
        container,
        SawtoothBinaryVariable(),
        C,
        names,
        alpha_levels,
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

    result = Dict{Tuple{String, Int}, JuMP.AffExpr}()

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

        # Binary variables α_1,...,α_L
        for j in alpha_levels
            alpha_container[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "SawtoothBin_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
        end

        # Linking constraint: g_0 = (x - x_min) / Δ
        link_container[name, t] = JuMP.@constraint(
            jump_model,
            g_container[name, 0, t] == (x_var - x_min) / delta,
        )

        # S^L constraints for j = 1,...,L
        for j in alpha_levels
            g_prev = g_container[name, j - 1, t]
            g_curr = g_container[name, j, t]
            alpha_j = alpha_container[name, j, t]

            # g_j ≤ 2 g_{j-1}
            JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev)
            # g_j ≤ 2(1 - g_{j-1})
            JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev))
            # g_j ≥ 2(g_{j-1} - α_j)
            JuMP.@constraint(jump_model, g_curr >= 2.0 * (g_prev - alpha_j))
            # g_j ≥ 2(α_j - g_{j-1})
            JuMP.@constraint(jump_model, g_curr >= 2.0 * (alpha_j - g_prev))
        end

        # Build x² ≈ x_min² + (2 x_min Δ + Δ²) g_0 - Σ_{j=1}^L Δ² 2^{-2j} g_j
        x_sq_approx = JuMP.AffExpr(x_min * x_min)
        JuMP.add_to_expression!(
            x_sq_approx,
            2.0 * x_min * delta + delta * delta,
            g_container[name, 0, t],
        )
        for j in alpha_levels
            coeff = delta * delta * (2.0^(-2 * j))
            JuMP.add_to_expression!(x_sq_approx, -coeff, g_container[name, j, t])
        end

        result[(name, t)] = x_sq_approx
    end

    return result
end
