# ZZI (Integer Zig-Zag) encoding for bivariate piecewise linear approximation of x·y.
# Builds the ZZI encoding from the Binary Reflected Gray Code (BRGC) to enforce SOS2
# adjacency on bivariate lambda variables using O(log₂ d) general integer variables.
# Includes triangle selection (6-stencil 3-coloring biclique cover) for strict SOS2 in 2D.
# Reference: Huchette & Vielma, "Nonconvex piecewise linear functions using polyhedral
# branching systems" (2023), Propositions 4–6 and Section 5.

"""
    build_zzi_encoding(d::Int) -> (Matrix{Int}, Matrix{Int})

Build the ZZI encoding matrix from the Binary Reflected Gray Code.

For `d` intervals:
- Computes `r = ceil(Int, log2(d))` and pads to `d_bar = 2^r`.
- Reuses `build_brgc(r)` for the Gray code `K` (d_bar × r matrix).
- Builds `C` (d × r): `C[i,k] = sum_{j=2}^{i} |K[j,k] - K[j-1,k]|` (cumulative transitions).
- Builds `C_ext` ((d+2) × r) with boundary conventions:
  row 1 = C_0 = C_1, rows 2..d+1 = C_1..C_d, row d+2 = C_{d+1} = C_d.

# Arguments
- `d::Int`: number of intervals (must be ≥ 1)

# Returns
- `(C, C_ext)`: the encoding matrix and its extended version with boundary rows.
"""
function build_zzi_encoding(d::Int)
    IS.@assert_op d >= 2
    r = ceil(Int, log2(d))
    d_bar = 2^r  # may be > d; K has d_bar rows, we use only the first d
    K = build_brgc(r)  # d_bar × r matrix

    # Build C (d × r): cumulative Gray code transitions
    C = Matrix{Int}(undef, d, r)
    for k in 1:r
        C[1, k] = 0  # C[1,k] = sum_{j=2}^{1}(...) = empty sum = 0
        for i in 2:d
            C[i, k] = C[i - 1, k] + abs(K[i, k] - K[i - 1, k])
        end
    end

    # Build C_ext ((d+2) × r):
    # row 1     = C_0 := C_1   (lower boundary convention)
    # rows 2..d+1 = C_1..C_d
    # row d+2   = C_{d+1} := C_d   (upper boundary convention)
    C_ext = Matrix{Int}(undef, d + 2, r)
    for k in 1:r
        C_ext[1, k] = C[1, k]           # C_0 = C_1
        for i in 1:d
            C_ext[i + 1, k] = C[i, k]   # rows 2..d+1 = C_1..C_d
        end
        C_ext[d + 2, k] = C[d, k]       # C_{d+1} = C_d
    end

    return C, C_ext
end

"General integer encoding variables y_k for ZZI bilinear approximation."
struct ZZIIntegerVariable <: VariableType end
"SOS2 encoding constraints linking lambda marginals and integer variables via ZZI branching."
struct ZZISOS2Constraint <: ConstraintType end

"""
    _add_zzi_sos2!(container, C, names, time_steps, lambda, d, axis_dim, other_dim_size, r, C_ext, meta)

Add ZZI integer-variable SOS2 encoding constraints for one axis of the bivariate lambda grid.

For each (name, t), creates `r` general integer variables (NOT binary) and adds 2r linear
constraints that enforce the SOS2 adjacency structure along the specified axis.

The integer variable `y_k` for level k has:
- `lower_bound = 0`
- `upper_bound = maximum(C_ext[:, k])`

The 2r constraints per (name, t) are:
- Lower: `sum_v C_ext[v, k] * marginal[v] <= y_k`   (v = 1..d+1, using C_{v-1,k})
- Upper: `y_k <= sum_v C_ext[v+1, k] * marginal[v]` (v = 1..d+1, using C_{v,k})

where `marginal[v]` sums lambda over the other dimension:
- axis_dim=1 (x-axis): `marginal[v] = sum_{w=1}^{other_dim_size} lambda[(name, v, w, t)]`
- axis_dim=2 (y-axis): `marginal[v] = sum_{w=1}^{other_dim_size} lambda[(name, w, v, t)]`

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `lambda`: Dict mapping `(name, i, j, t)` tuples to `JuMP.VariableRef`
- `d::Int`: number of intervals along this axis
- `axis_dim::Int`: 1 for x-axis, 2 for y-axis
- `other_dim_size::Int`: size of the other axis (d2+1 for x-axis, d1+1 for y-axis)
- `r::Int`: encoding depth = `ceil(Int, log2(d))`
- `C_ext::Matrix{Int}`: extended ZZI encoding matrix of size (d+2) × r
- `meta::String`: identifier encoding the axis label for variable/constraint names
"""
function _add_zzi_sos2!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    lambda,
    d::Int,
    axis_dim::Int,
    other_dim_size::Int,
    r::Int,
    C_ext::Matrix{Int},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op axis_dim in (1, 2)
    IS.@assert_op size(C_ext, 1) == d + 2
    IS.@assert_op size(C_ext, 2) == r

    jump_model = get_jump_model(container)

    y_container = add_variable_container!(
        container,
        ZZIIntegerVariable(),
        C,
        names,
        1:r,
        time_steps;
        meta,
    )
    enc_cons = add_constraints_container!(
        container,
        ZZISOS2Constraint(),
        C,
        names,
        1:(2 * r),
        time_steps;
        meta,
    )

    # d+1 marginal points along this axis (breakpoints v = 1..d+1)
    n_marginals = d + 1

    for name in names, t in time_steps
        # Build marginals: sum of lambda over the other dimension for each breakpoint v
        # v indexes the current axis (1..d+1); w indexes the other axis (1..other_dim_size)
        marginals = Vector{JuMP.AffExpr}(undef, n_marginals)
        for v in 1:n_marginals
            m = JuMP.AffExpr(0.0)
            for w in 1:other_dim_size
                if axis_dim == 1
                    lam = lambda[(name, v, w, t)]
                else
                    lam = lambda[(name, w, v, t)]
                end
                add_proportional_to_jump_expression!(m, lam, 1.0)
            end
            marginals[v] = m
        end

        for k in 1:r
            # Compute upper bound for y_k from C_ext column k
            y_ub = maximum(C_ext[v + 1, k] for v in 1:n_marginals)

            y_k = y_container[name, k, t] = JuMP.@variable(
                jump_model,
                base_name = "ZZI_y_$(meta)_$(C)_{$(name), $(k), $(t)}",
                lower_bound = 0,
                upper_bound = y_ub,
                integer = true,
            )

            # Lower encoding constraint: sum_v C_ext[v, k] * marginal[v] <= y_k
            # C_ext[v, k] corresponds to C_{v-1, k} in math (0-indexed C_0..C_d)
            lower_lhs = JuMP.AffExpr(0.0)
            for v in 1:n_marginals
                coeff = C_ext[v, k]  # row v = C_{v-1} in math notation
                if coeff != 0
                    add_proportional_to_jump_expression!(
                        lower_lhs, marginals[v], Float64(coeff),
                    )
                end
            end
            enc_cons[name, 2 * k - 1, t] =
                JuMP.@constraint(jump_model, lower_lhs <= y_k)

            # Upper encoding constraint: y_k <= sum_v C_ext[v+1, k] * marginal[v]
            # C_ext[v+1, k] corresponds to C_{v, k} in math notation
            upper_rhs = JuMP.AffExpr(0.0)
            for v in 1:n_marginals
                coeff = C_ext[v + 1, k]  # row v+1 = C_v in math notation
                if coeff != 0
                    add_proportional_to_jump_expression!(
                        upper_rhs, marginals[v], Float64(coeff),
                    )
                end
            end
            enc_cons[name, 2 * k, t] =
                JuMP.@constraint(jump_model, y_k <= upper_rhs)
        end
    end

    return
end

"""
    _choose_triangulation(x_bkpts, y_bkpts) -> Matrix{Symbol}

Choose the diagonal triangulation for each subrectangle of the bivariate grid.

For each subrectangle (i, j) (i = 1..d1, j = 1..d2) with corners
`(x_bkpts[i], y_bkpts[j])` to `(x_bkpts[i+1], y_bkpts[j+1])`, picks the diagonal
that minimizes the maximum interpolation error at the center of the rectangle.

- `:U` diagonal connects `(i,j)−(i+1,j+1)`: interpolant at center =
  `0.5*(x_lo*y_lo + x_hi*y_hi)`
- `:K` diagonal connects `(i+1,j)−(i,j+1)`: interpolant at center =
  `0.5*(x_hi*y_lo + x_lo*y_hi)`

The diagonal whose interpolant is closer to `x_mid * y_mid` is chosen.

# Arguments
- `x_bkpts`: breakpoints along the x-axis (length d1+1)
- `y_bkpts`: breakpoints along the y-axis (length d2+1)

# Returns
- `Matrix{Symbol}` of size d1 × d2 with entries `:U` or `:K`.
"""
function _choose_triangulation(x_bkpts, y_bkpts)
    d1 = length(x_bkpts) - 1
    d2 = length(y_bkpts) - 1
    IS.@assert_op d1 >= 1
    IS.@assert_op d2 >= 1

    triang = Matrix{Symbol}(undef, d1, d2)

    for i in 1:d1, j in 1:d2
        x_lo = x_bkpts[i]
        x_hi = x_bkpts[i + 1]
        y_lo = y_bkpts[j]
        y_hi = y_bkpts[j + 1]

        x_mid = 0.5 * (x_lo + x_hi)
        y_mid = 0.5 * (y_lo + y_hi)
        exact = x_mid * y_mid

        # :U diagonal (connects (i,j) to (i+1,j+1))
        interp_U = 0.5 * (x_lo * y_lo + x_hi * y_hi)

        # :K diagonal (connects (i+1,j) to (i,j+1))
        interp_K = 0.5 * (x_hi * y_lo + x_lo * y_hi)

        triang[i, j] = abs(interp_U - exact) <= abs(interp_K - exact) ? :U : :K
    end

    return triang
end

"Binary triangle-selection variables w_s ∈ {0,1} for ZZI bilinear approximation."
struct ZZITriangleVariable <: VariableType end
"Biclique cover constraints linking triangle selections to bivariate lambda variables."
struct ZZITriangleConstraint <: ConstraintType end

"""
    _add_triangle_selection!(container, C, names, time_steps, lambda, d1, d2, triang, meta)

Add the 6-stencil biclique cover triangle selection constraints for the bivariate ZZI formulation.

For each (name, t), adds 6 binary variables `w[1..6]` and 12 linear constraints that enforce
the proper triangle adjacency structure on the bivariate lambda grid via a 3-coloring biclique cover.

The biclique cover is split into two groups of 3:

**Diagonal bicliques (indices 1–3, for `:U` subrectangles):**
For tau = 0, 1, 2, iterates subrectangles (si, sj) with `triang[si, sj] == :U`.
The conflict pair is `(i1, j1) = (si+1, sj)` vs `(i2, j2) = (si, sj+1)`.
If `(i1 + j1) % 3 == tau`, vertex (i1, j1) goes into A_sum and (i2, j2) into B_sum.
Constraints: `A_sum <= w[tau+1]` and `B_sum <= 1 - w[tau+1]`.

**Anti-diagonal bicliques (indices 4–6, for `:K` subrectangles):**
For tau = 0, 1, 2, iterates subrectangles (si, sj) with `triang[si, sj] == :K`.
The conflict pair is `(i1, j1) = (si, sj)` vs `(i2, j2) = (si+1, sj+1)`.
If `(i1 + j1) % 3 == tau`, vertex (i1, j1) goes into A_sum and (i2, j2) into B_sum.
Constraints: `A_sum <= w[tau+4]` and `B_sum <= 1 - w[tau+4]`.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `lambda`: Dict mapping `(name, i, j, t)` tuples to `JuMP.VariableRef`
- `d1::Int`: number of intervals along the x-axis
- `d2::Int`: number of intervals along the y-axis
- `triang::Matrix{Symbol}`: triangulation matrix of size d1 × d2 (`:U` or `:K` per cell)
- `meta::String`: identifier for variable/constraint naming
"""
function _add_triangle_selection!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    lambda,
    d1::Int,
    d2::Int,
    triang::Matrix{Symbol},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op size(triang) == (d1, d2)

    jump_model = get_jump_model(container)

    w_container = add_variable_container!(
        container,
        ZZITriangleVariable(),
        C,
        names,
        1:6,
        time_steps;
        meta,
    )
    tri_cons = add_constraints_container!(
        container,
        ZZITriangleConstraint(),
        C,
        names,
        1:12,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        # Create 6 binary triangle-selection variables
        w = Vector{JuMP.VariableRef}(undef, 6)
        for s in 1:6
            w[s] = w_container[name, s, t] = JuMP.@variable(
                jump_model,
                base_name = "ZZI_w_$(C)_{$(name), $(s), $(t)}",
                binary = true,
            )
        end

        # Diagonal bicliques (indices 1–3): conflict edges from :U subrectangles
        for tau in 0:2
            idx = tau + 1  # constraint index: 1, 2, 3
            A_sum = JuMP.AffExpr(0.0)
            B_sum = JuMP.AffExpr(0.0)

            for si in 1:d1, sj in 1:d2
                if triang[si, sj] == :U
                    # Conflict pair for :U cell (si, sj):
                    # vertex (i1, j1) = (si+1, sj) vs (i2, j2) = (si, sj+1)
                    i1, j1 = si + 1, sj
                    i2, j2 = si, sj + 1
                    if (i1 + j1) % 3 == tau
                        lam_A = lambda[(name, i1, j1, t)]
                        lam_B = lambda[(name, i2, j2, t)]
                        add_proportional_to_jump_expression!(A_sum, lam_A, 1.0)
                        add_proportional_to_jump_expression!(B_sum, lam_B, 1.0)
                    end
                end
            end

            # A_sum <= w[idx]
            tri_cons[name, 2 * tau + 1, t] =
                JuMP.@constraint(jump_model, A_sum <= w[idx])
            # B_sum <= 1 - w[idx]
            tri_cons[name, 2 * tau + 2, t] =
                JuMP.@constraint(jump_model, B_sum <= 1 - w[idx])
        end

        # Anti-diagonal bicliques (indices 4–6): conflict edges from :K subrectangles
        for tau in 0:2
            idx = tau + 4  # constraint index: 4, 5, 6 → constraint rows 7–12
            A_sum = JuMP.AffExpr(0.0)
            B_sum = JuMP.AffExpr(0.0)

            for si in 1:d1, sj in 1:d2
                if triang[si, sj] == :K
                    # Conflict pair for :K cell (si, sj):
                    # vertex (i1, j1) = (si, sj) vs (i2, j2) = (si+1, sj+1)
                    i1, j1 = si, sj
                    i2, j2 = si + 1, sj + 1
                    if (i1 + j1) % 3 == tau
                        lam_A = lambda[(name, i1, j1, t)]
                        lam_B = lambda[(name, i2, j2, t)]
                        add_proportional_to_jump_expression!(A_sum, lam_A, 1.0)
                        add_proportional_to_jump_expression!(B_sum, lam_B, 1.0)
                    end
                end
            end

            # A_sum <= w[idx]
            tri_cons[name, 2 * tau + 7, t] =
                JuMP.@constraint(jump_model, A_sum <= w[idx])
            # B_sum <= 1 - w[idx]
            tri_cons[name, 2 * tau + 8, t] =
                JuMP.@constraint(jump_model, B_sum <= 1 - w[idx])
        end
    end

    return
end
