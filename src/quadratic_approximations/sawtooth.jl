# Sawtooth MIP approximation of x² for use in constraints.
# Uses recursive tooth function compositions with O(log(1/ε)) binary variables.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024).

"Auxiliary continuous variables (g₀, …, g_L) for sawtooth quadratic approximation."
struct SawtoothAuxVariable <: VariableType end
"Binary variables (α₁, …, α_L) for sawtooth quadratic approximation."
struct SawtoothBinaryVariable <: VariableType end
"Links g₀ to the normalized x value in sawtooth quadratic approximation."
struct SawtoothLinkingConstraint <: ConstraintType end
"Constrains g_j based on g_{j-1}."
struct SawtoothMIPConstraint <: ConstraintType end
struct SawtoothLPConstraint <: ConstraintType end

"""
    _add_sawtooth_quadratic_approx!(container, C, names, time_steps, x_var, x_min, x_max, depth, meta)

Approximate x² using the sawtooth MIP formulation.

Creates auxiliary continuous variables g_0,...,g_L and binary variables α_1,...,α_L,
adds S^L constraints (4 per level) and a linking constraint for each component and
time step, and stores affine expressions approximating x² in a
`QuadraticApproxExpression` expression container.

For depth L, the approximation interpolates x² at 2^L + 1 uniformly spaced breakpoints
with maximum overestimation error Δ² · 2^{-2L-2} where Δ = x_max - x_min.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `depth::Int`: sawtooth depth L (number of binary variables per component per time step)
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function _add_sawtooth_quadratic_approx!(
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

    # Create containers with known dimensions
    g_levels = 0:depth
    alpha_levels = 1:depth
    g_var = @_add_container!(variable, SawtoothAuxVariable, g_levels)
    alpha_var = @_add_container!(variable, SawtoothBinaryVariable, alpha_levels)
    mip_cons = @_add_container!(constraints, SawtoothMIPConstraint, 1:4, sparse)
    link_cons = @_add_container!(constraints, SawtoothLinkingConstraint)
    result_expr = @_add_container!(expression, QuadraticApproxExpression)

    # Precompute sawtooth coefficients (invariant across names and time steps)
    saw_coeffs = [delta * delta * (2.0^(-2 * j)) for j in alpha_levels]

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

        # Binary variables α_1,...,α_L
        for j in alpha_levels
            alpha_var[name, j, t] = JuMP.@variable(
                jump_model,
                base_name = "SawtoothBin_$(C)_{$(name), $(j), $(t)}",
                binary = true,
            )
        end

        # Linking constraint: g_0 = (x - x_min) / Δ
        link_cons[name, t] = JuMP.@constraint(
            jump_model,
            g_var[name, 0, t] == (x - x_min) / delta,
        )

        # S^L constraints for j = 1,...,L
        for j in alpha_levels
            g_prev = g_var[name, j - 1, t]
            g_curr = g_var[name, j, t]
            alpha_j = alpha_var[name, j, t]

            # g_j ≤ 2 g_{j-1}
            mip_cons[name, 1, t] = JuMP.@constraint(jump_model, g_curr <= 2.0 * g_prev)
            # g_j ≤ 2(1 - g_{j-1})
            mip_cons[name, 2, t] =
                JuMP.@constraint(jump_model, g_curr <= 2.0 * (1.0 - g_prev))
            # g_j ≥ 2(g_{j-1} - α_j)
            mip_cons[name, 3, t] =
                JuMP.@constraint(jump_model, g_curr >= 2.0 * (g_prev - alpha_j))
            # g_j ≥ 2(α_j - g_{j-1})
            mip_cons[name, 4, t] =
                JuMP.@constraint(jump_model, g_curr >= 2.0 * (alpha_j - g_prev))
        end

        # Build x² ≈ x_min² + (2 x_min Δ + Δ²) g_0 - Σ_{j=1}^L Δ² 2^{-2j} g_j
        x_sq_approx = JuMP.AffExpr(x_min * x_min)
        JuMP.add_to_expression!(
            x_sq_approx,
            2.0 * x_min * delta + delta * delta,
            g_var[name, 0, t],
        )
        for j in alpha_levels
            JuMP.add_to_expression!(x_sq_approx, -saw_coeffs[j], g_var[name, j, t])
        end

        result_expr[name, t] = x_sq_approx
    end

    return result_expr
end
