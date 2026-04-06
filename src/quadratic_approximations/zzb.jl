# ZZB (Binary Zig-Zag) encoding for piecewise linear quadratic approximation of x².
# Uses binary reflected Gray code to construct an independent branching system that enforces
# SOS2 adjacency with O(log₂ d) binary variables for d breakpoint intervals.
# Reference: Huchette & Vielma, "Nonconvex piecewise linear functions using polyhedral
# branching systems" (2023), Proposition 3.

"Lambda (convex combination weight) variables for ZZB quadratic approximation."
struct ZZBLambdaVariable <: SparseVariableType end
"Binary encoding variables y ∈ {0,1}^r for ZZB quadratic approximation."
struct ZZBBinaryVariable <: VariableType end
"Encoding constraints linking lambda and binary variables via zig-zag branching."
struct ZZBEncodingConstraint <: ConstraintType end
"Tightened variable z bounded between epigraph lower and ZZB upper bounds."
struct ZZBTightenedVariable <: VariableType end
"Constraints bounding the tightened variable (z ≤ upper, z ≥ lower)."
struct ZZBTightenedConstraint <: ConstraintType end

"""
Config for ZZB (Binary Zig-Zag) quadratic approximation.

# Fields
- `depth::Int`: number of PWL segments; must be ≥ 2 and a power of 2 (log₂(depth) binary variables are used)
- `epigraph_depth::Int`: LP tightening depth via epigraph Q^{L1} lower bound; 0 to disable (default 0)
"""
struct ZZBQuadConfig <: QuadraticApproxConfig
    depth::Int
    epigraph_depth::Int
end
ZZBQuadConfig(depth::Int) = ZZBQuadConfig(depth, 0)

"""
    build_brgc(r::Int) -> Matrix{Int}

Build the Binary Reflected Gray Code (BRGC) matrix of depth r.

Returns a 2^r × r matrix where each row is a Gray code word.
For r=1, G=[0; 1]. For r>1, take G_{r-1}, reflect vertically,
prepend 0s to the top half and 1s to the bottom half.
"""
function build_brgc(r::Int)
    IS.@assert_op r >= 1
    if r == 1
        return [0; 1][:, :]  # 2×1 matrix
    end
    G_prev = build_brgc(r - 1)
    n = size(G_prev, 1)
    G = Matrix{Int}(undef, 2 * n, r)
    # Top half: prepend 0, keep order
    for i in 1:n
        G[i, 1] = 0
        for k in 1:(r - 1)
            G[i, k + 1] = G_prev[i, k]
        end
    end
    # Bottom half: prepend 1, reverse order
    for i in 1:n
        G[n + i, 1] = 1
        for k in 1:(r - 1)
            G[n + i, k + 1] = G_prev[n - i + 1, k]
        end
    end
    return G
end

"""
    _build_zzb_coefficients(G::Matrix{Int}, depth::Int) -> (Matrix{Int}, Matrix{Int})

Build lower and upper coefficient matrices for ZZB encoding constraints.

Uses independent branching: for each level k and breakpoint i, the coefficient is
the min/max of G[j, k] over segments adjacent to breakpoint i. This ensures each
encoding constraint involves only the single binary variable y_k (no coupling).

Returns (lower_coeffs, upper_coeffs), each of size (2^depth + 1) × depth.
"""
function _build_zzb_coefficients(G::Matrix{Int}, depth::Int)
    d = size(G, 1)          # 2^depth
    n_points = d + 1
    lower_coeffs = Matrix{Int}(undef, n_points, depth)
    upper_coeffs = Matrix{Int}(undef, n_points, depth)

    for k in 1:depth
        # First breakpoint: adjacent to segment 1 only
        lower_coeffs[1, k] = G[1, k]
        upper_coeffs[1, k] = G[1, k]

        # Interior breakpoints: adjacent to segments i-1 and i
        for i in 2:d
            lower_coeffs[i, k] = min(G[i - 1, k], G[i, k])
            upper_coeffs[i, k] = max(G[i - 1, k], G[i, k])
        end

        # Last breakpoint: adjacent to segment d only
        lower_coeffs[n_points, k] = G[d, k]
        upper_coeffs[n_points, k] = G[d, k]
    end

    return lower_coeffs, upper_coeffs
end

"""
    _add_quadratic_approx!(config::ZZBQuadConfig, container, C, names, time_steps, x_var, x_min, x_max, meta)

Approximate x² using the ZZB (Binary Zig-Zag) encoding formulation.

Creates lambda (λ) convex combination variables over config.depth + 1 breakpoints,
log₂(config.depth) binary encoding variables, adds linking, normalization, and
zig-zag encoding constraints, and stores affine expressions approximating x²
in a `QuadraticExpression` expression container.

The encoding constraints use independent branching: each constraint bounds
y_k between min and max of G[j, k] over segments adjacent to each breakpoint.

# Arguments
- `config::ZZBQuadConfig`: configuration with `depth` (number of PWL segments; must be ≥ 2, power of 2) and `epigraph_depth` (LP tightening depth; 0 to disable)
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
    config::ZZBQuadConfig,
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
    IS.@assert_op config.depth >= 2

    depth = round(Int, log2(config.depth))
    d = 2^depth
    n_points = d + 1
    x_bkpts, x_sq_bkpts =
        _get_breakpoints_for_pwl_function(x_min, x_max, _square; num_segments = d)
    jump_model = get_jump_model(container)

    # Build encoding coefficient matrices once (invariant across names and time steps)
    G = build_brgc(depth)
    lower_coeffs, upper_coeffs = _build_zzb_coefficients(G, depth)

    # Create all containers upfront
    lambda_container =
        add_variable_container!(container, ZZBLambdaVariable(), C; meta)
    y_container = add_variable_container!(
        container,
        ZZBBinaryVariable(),
        C,
        names,
        1:depth,
        time_steps;
        meta,
    )
    link_cons = add_constraints_container!(
        container,
        SOS2LinkingConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    link_expr = add_expression_container!(
        container,
        SOS2LinkingExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    norm_cons = add_constraints_container!(
        container,
        SOS2NormConstraint(),
        C,
        names,
        time_steps;
        meta,
    )
    norm_expr = add_expression_container!(
        container,
        SOS2NormExpression(),
        C,
        names,
        time_steps;
        meta,
    )
    enc_cons = add_constraints_container!(
        container,
        ZZBEncodingConstraint(),
        C,
        names,
        1:(2 * depth),
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

    if config.epigraph_depth > 0
        lp_expr = _add_quadratic_approx!(
            EpigraphQuadConfig(config.epigraph_depth),
            container, C, names, time_steps,
            x_var, x_min, x_max, meta * "_lb",
        )
        z_var = add_variable_container!(
            container,
            ZZBTightenedVariable(),
            C,
            names,
            time_steps;
            meta,
        )
        tight_cons = add_constraints_container!(
            container,
            ZZBTightenedConstraint(),
            C,
            names,
            1:2,
            time_steps;
            meta,
        )
    end

    # Compute valid bounds for z ≈ x²
    z_min = (x_min <= 0.0 <= x_max) ? 0.0 : min(x_min * x_min, x_max * x_max)
    z_max = max(x_min * x_min, x_max * x_max)

    for name in names, t in time_steps
        x = x_var[name, t]

        # Create lambda variables: λ_i ∈ [0, 1]
        lambda = Vector{JuMP.VariableRef}(undef, n_points)
        for i in 1:n_points
            lambda[i] =
                lambda_container[(name, i, t)] = JuMP.@variable(
                    jump_model,
                    base_name = "ZZBLambda_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = 1.0,
                )
        end

        # x = Σ λ_i * x_bkpts_i (linking constraint)
        link = link_expr[name, t] = JuMP.AffExpr(0.0)
        for i in eachindex(x_bkpts)
            add_proportional_to_jump_expression!(link, lambda[i], x_bkpts[i])
        end
        link_cons[name, t] = JuMP.@constraint(jump_model, x == link)

        # Σ λ_i = 1 (normalization)
        norm = norm_expr[name, t] = JuMP.AffExpr(0.0)
        for l in lambda
            add_proportional_to_jump_expression!(norm, l, 1.0)
        end
        norm_cons[name, t] = JuMP.@constraint(jump_model, norm == 1.0)

        # Create binary encoding variables y_k ∈ {0,1} for k=1..depth
        y_vars = Vector{JuMP.VariableRef}(undef, depth)
        for k in 1:depth
            y_vars[k] =
                y_container[name, k, t] = JuMP.@variable(
                    jump_model,
                    base_name = "ZZBBinary_$(C)_{$(name), $(k), $(t)}",
                    binary = true,
                )
        end

        # Encoding constraints using independent branching.
        # For each level k, bound y_k between min and max of G[j,k]
        # over segments adjacent to each breakpoint.
        for k in 1:depth
            # Lower encoding constraint: Σ lower_coeffs[i,k] * λ[i] ≤ y_k
            lower_lhs = JuMP.AffExpr(0.0)
            for i in 1:n_points
                coeff = lower_coeffs[i, k]
                if coeff != 0
                    add_proportional_to_jump_expression!(
                        lower_lhs, lambda[i], Float64(coeff),
                    )
                end
            end
            enc_cons[name, 2 * k - 1, t] =
                JuMP.@constraint(jump_model, lower_lhs <= y_vars[k])

            # Upper encoding constraint: y_k ≤ Σ upper_coeffs[i,k] * λ[i]
            upper_rhs = JuMP.AffExpr(0.0)
            for i in 1:n_points
                coeff = upper_coeffs[i, k]
                if coeff != 0
                    add_proportional_to_jump_expression!(
                        upper_rhs, lambda[i], Float64(coeff),
                    )
                end
            end
            enc_cons[name, 2 * k, t] =
                JuMP.@constraint(jump_model, y_vars[k] <= upper_rhs)
        end

        # Build x̂² = Σ λ_i * x_bkpts_i² as an affine expression (upper bound on x²)
        x_sq_upper = JuMP.AffExpr(0.0)
        for i in 1:n_points
            add_proportional_to_jump_expression!(x_sq_upper, lambda[i], x_sq_bkpts[i])
        end

        if config.epigraph_depth > 0
            z =
                z_var[name, t] = JuMP.@variable(
                    jump_model,
                    base_name = "TightenedZZB_$(C)_{$(name), $(t)}",
                    lower_bound = z_min,
                    upper_bound = z_max,
                )
            tight_cons[name, 1, t] = JuMP.@constraint(jump_model, z <= x_sq_upper)
            tight_cons[name, 2, t] = JuMP.@constraint(jump_model, z >= lp_expr[name, t])
            result_expr[name, t] = JuMP.AffExpr(0.0, z => 1.0)
        else
            result_expr[name, t] = x_sq_upper
        end
    end

    return result_expr
end
