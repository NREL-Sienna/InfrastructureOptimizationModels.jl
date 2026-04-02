# Sawtooth MIP approximation of x² for use in constraints.
# Uses recursive tooth function compositions with O(log(1/ε)) binary variables.
# Reference: Beach, Burlacu, Hager, Hildebrand (2024).

"Auxiliary continuous variables (g₀, …, g_L) for sawtooth quadratic approximation."
struct SawtoothAuxVariable <: VariableType end
"Binary variables (α₁, …, α_L) for sawtooth quadratic approximation."
struct SawtoothBinaryVariable <: VariableType end
"Variable result in tightened version."
struct SawtoothTightenedVariable <: VariableType end
"Links g₀ to the normalized x value in sawtooth quadratic approximation."
struct SawtoothLinkingConstraint <: ConstraintType end
"Constrains g_j based on g_{j-1}."
struct SawtoothMIPConstraint <: ConstraintType end
struct SawtoothLPConstraint <: ConstraintType end
"Bounds tightened variable."
struct SawtoothTightenedConstraint <: ConstraintType end

"Config for sawtooth MIP quadratic approximation."
struct SawtoothQuadConfig <: QuadraticApproxConfig
    depth::Int
    tighten::Bool
end
SawtoothQuadConfig(depth::Int) = SawtoothQuadConfig(depth, false)

"""
    _add_quadratic_approx!(config::SawtoothQuadConfig, container, C, names, time_steps, x_var, x_min, x_max, meta)

Approximate x² using the sawtooth MIP formulation.

Creates auxiliary continuous variables g_0,...,g_L and binary variables α_1,...,α_L,
adds S^L constraints (4 per level) and a linking constraint for each component and
time step, and stores affine expressions approximating x² in a
`QuadraticExpression` expression container.

For depth L, the approximation interpolates x² at 2^L + 1 uniformly spaced breakpoints
with maximum overestimation error Δ² · 2^{-2L-2} where Δ = x_max - x_min.

# Arguments
- `config::SawtoothQuadConfig`: configuration
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `meta::String`: variable type identifier for the approximation (allows multiple approximations per component type)
"""
function _add_quadratic_approx!(
    config::SawtoothQuadConfig,
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op config.depth >= 1
    epigraph_depth = max(2, ceil(Int, 1.5 * config.depth))
    jump_model = get_jump_model(container)
    delta = x_max - x_min

    # Create containers with known dimensions
    g_levels = 0:(config.depth)
    alpha_levels = 1:(config.depth)
    g_var = add_variable_container!(
        container,
        SawtoothAuxVariable(),
        C,
        names,
        g_levels,
        time_steps;
        meta,
    )
    alpha_var = add_variable_container!(
        container,
        SawtoothBinaryVariable(),
        C,
        names,
        alpha_levels,
        time_steps;
        meta,
    )
    mip_cons = add_constraints_container!(
        container,
        SawtoothMIPConstraint(),
        C,
        names,
        1:4,
        time_steps;
        sparse = true,
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        SawtoothLinkingConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    result_expr = add_expression_container!(
        container,
        QuadraticExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    if config.tighten
        lp_expr = _add_quadratic_approx!(
            EpigraphQuadConfig(), container, C, names, time_steps,
            x_var, x_min, x_max,
            epigraph_depth, meta * "_lb",
        )
        z_var = add_variable_container!(
            container,
            SawtoothTightenedVariable(),
            C,
            names,
            time_steps;
            meta,
        )
        tight_cons = add_constraints_container!(
            container,
            SawtoothTightenedConstraint(),
            C,
            names,
            1:2,
            time_steps;
            meta,
        )
    end

    # Precompute sawtooth coefficients (invariant across names and time steps)
    saw_coeffs = [delta * delta * (2.0^(-2 * j)) for j in alpha_levels]

    # Compute valid bounds for z ≈ x² from variable bounds
    z_min = (x_min <= 0.0 <= x_max) ? 0.0 : min(x_min * x_min, x_max * x_max)
    z_max = max(x_min * x_min, x_max * x_max)

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
        add_proportional_to_jump_expression!(
            x_sq_approx,
            g_var[name, 0, t],
            2.0 * x_min * delta + delta * delta,
        )
        for j in alpha_levels
            add_proportional_to_jump_expression!(
                x_sq_approx,
                g_var[name, j, t],
                -saw_coeffs[j],
            )
        end

        if config.tighten
            z =
                z_var[name, t] = JuMP.@variable(
                    jump_model,
                    base_name = "TightenedSawtooth_$(C)_{$(name), $(t)}",
                    lower_bound = z_min,
                    upper_bound = z_max
                )
            tight_cons[name, 1, t] = JuMP.@constraint(jump_model, z <= x_sq_approx)
            tight_cons[name, 2, t] = JuMP.@constraint(jump_model, z >= lp_expr[name, t])
            result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
        else
            result_expr[name, t] = x_sq_approx
        end
    end

    return result_expr
end
