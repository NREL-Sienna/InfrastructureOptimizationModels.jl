# Piecewise McCormick (PWMCC) cuts for concave terms in Bin2 bilinear approximation.
# Adds K local chord upper bounds on the v^2 SoS2 approximation by partitioning
# each concave term's domain into K sub-intervals.
# LP gap shrinks from Delta^2/4 to Delta^2/(4K^2).
# These cuts supplement (do not replace) existing SoS2 constraints.

"Binary interval selector for piecewise McCormick cuts."
struct PiecewiseMcCormickBinary <: SparseVariableType end

"Disaggregated variable for piecewise McCormick cuts."
struct PiecewiseMcCormickDisaggregated <: SparseVariableType end

"Selector sum constraint: sum_k delta_k = 1."
struct PiecewiseMcCormickSelectorSum <: ConstraintType end

"Disaggregation linking constraint: v = sum_k v^d_k."
struct PiecewiseMcCormickLinking <: ConstraintType end

"Interval activation lower bound: t_{k-1} * delta_k <= v^d_k."
struct PiecewiseMcCormickIntervalLB <: ConstraintType end

"Interval activation upper bound: v^d_k <= t_k * delta_k."
struct PiecewiseMcCormickIntervalUB <: ConstraintType end

"Piecewise McCormick chord upper-bound constraint on v^2 approximation."
struct PiecewiseMcCormickChordUB <: ConstraintType end

"Piecewise McCormick tangent lower-bound constraint (left endpoint)."
struct PiecewiseMcCormickTangentLBL <: ConstraintType end

"Piecewise McCormick tangent lower-bound constraint (right endpoint)."
struct PiecewiseMcCormickTangentLBR <: ConstraintType end

"""
    _add_pwmcc_concave_cuts!(jump_model, v, q, bounds, K, meta)

Add piecewise McCormick cuts on a concave term (-v²) for a single (name, t) pair.

Partitions [bounds.min, bounds.max] into K uniform sub-intervals and creates
disaggregated variables, binary interval selectors, and chord/tangent constraints
that cut off the interior of the SoS2 relaxation polytope.

# Arguments
- `jump_model::JuMP.Model`: the JuMP model to add variables and constraints to
- `v::JuMP.AbstractJuMPScalar`: the original variable for this (name, t) pair
- `q::JuMP.AbstractJuMPScalar`: the SoS2 approximation expression for v² at this (name, t)
- `bounds::MinMax`: named tuple `(min, max)` giving the domain of `v`
- `K::Int`: number of sub-intervals (K ≥ 1; K=2 is the minimal useful choice)
- `meta::String`: base name prefix for JuMP variables, e.g. `"pwmcc_x_gen1_3"`

# Returns
A `NamedTuple` with fields:
- `delta_vars::Vector{JuMP.VariableRef}` – K binary interval selectors
- `vd_vars::Vector{JuMP.VariableRef}` – K disaggregated variables
- `selector_con::JuMP.ConstraintRef` – selector sum constraint (∑ δ_k = 1)
- `linking_con::JuMP.ConstraintRef` – linking constraint (∑ vd_k = v)
- `interval_lb_cons::Vector{JuMP.ConstraintRef}` – K interval lower-bound constraints
- `interval_ub_cons::Vector{JuMP.ConstraintRef}` – K interval upper-bound constraints
- `chord_ub_con::JuMP.ConstraintRef` – piecewise chord upper bound on q
- `tangent_lb_l_con::JuMP.ConstraintRef` – tangent lower bound (left endpoints)
- `tangent_lb_r_con::JuMP.ConstraintRef` – tangent lower bound (right endpoints)
"""
function _add_pwmcc_concave_cuts!(
    jump_model::JuMP.Model,
    v::JuMP.AbstractJuMPScalar,
    q::JuMP.AbstractJuMPScalar,
    bounds::MinMax,
    K::Int,
    meta::String,
)
    IS.@assert_op K >= 1
    IS.@assert_op bounds.min < bounds.max

    v_min = bounds.min
    v_max = bounds.max

    # Pre-compute breakpoints and derived coefficients
    brk = [v_min + k * (v_max - v_min) / K for k in 0:K]
    sum_brk = [brk[k] + brk[k + 1] for k in 1:K]
    prod_brk = [brk[k] * brk[k + 1] for k in 1:K]
    two_brk_l = [2.0 * brk[k] for k in 1:K]
    sq_brk_l = [brk[k]^2 for k in 1:K]
    two_brk_r = [2.0 * brk[k + 1] for k in 1:K]
    sq_brk_r = [brk[k + 1]^2 for k in 1:K]

    # Create binary selectors and disaggregated variables
    delta = Vector{JuMP.VariableRef}(undef, K)
    vd = Vector{JuMP.VariableRef}(undef, K)
    for k in 1:K
        delta[k] = JuMP.@variable(
            jump_model,
            base_name = "PwMcCBin_$(meta)_$(k)",
            binary = true,
        )
        vd[k] = JuMP.@variable(
            jump_model,
            base_name = "PwMcCDis_$(meta)_$(k)",
        )
    end

    # Selector sum: sum_k delta_k = 1
    sel_expr = JuMP.AffExpr(0.0)
    for k in 1:K
        JuMP.add_to_expression!(sel_expr, delta[k])
    end
    sel_con = JuMP.@constraint(jump_model, sel_expr == 1.0)

    # Linking: sum_k vd_k = v
    link_expr = JuMP.AffExpr(0.0)
    for k in 1:K
        JuMP.add_to_expression!(link_expr, vd[k])
    end
    link_con = JuMP.@constraint(jump_model, link_expr == v)

    # Interval activation bounds
    ilb = Vector{JuMP.ConstraintRef}(undef, K)
    iub = Vector{JuMP.ConstraintRef}(undef, K)
    for k in 1:K
        ilb[k] = JuMP.@constraint(jump_model, brk[k] * delta[k] <= vd[k])
        iub[k] = JuMP.@constraint(jump_model, vd[k] <= brk[k + 1] * delta[k])
    end

    # Chord upper bound: prevents q from exceeding the local piecewise chord
    # of v^2 in the LP relaxation (tightens from global chord to piecewise).
    chord_rhs = JuMP.AffExpr(0.0)
    for k in 1:K
        JuMP.add_to_expression!(chord_rhs, sum_brk[k], vd[k])
        JuMP.add_to_expression!(chord_rhs, -prod_brk[k], delta[k])
    end
    chord_con = JuMP.@constraint(jump_model, q <= chord_rhs)

    # Tangent lower bounds from convexity of v^2 at interval endpoints.
    tang_l_rhs = JuMP.AffExpr(0.0)
    for k in 1:K
        JuMP.add_to_expression!(tang_l_rhs, two_brk_l[k], vd[k])
        JuMP.add_to_expression!(tang_l_rhs, -sq_brk_l[k], delta[k])
    end
    tll = JuMP.@constraint(jump_model, q >= tang_l_rhs)

    tang_r_rhs = JuMP.AffExpr(0.0)
    for k in 1:K
        JuMP.add_to_expression!(tang_r_rhs, two_brk_r[k], vd[k])
        JuMP.add_to_expression!(tang_r_rhs, -sq_brk_r[k], delta[k])
    end
    tlr = JuMP.@constraint(jump_model, q >= tang_r_rhs)

    return (
        delta_vars = delta,
        vd_vars = vd,
        selector_con = sel_con,
        linking_con = link_con,
        interval_lb_cons = ilb,
        interval_ub_cons = iub,
        chord_ub_con = chord_con,
        tangent_lb_l_con = tll,
        tangent_lb_r_con = tlr,
    )
end

"""
    _create_pwmcc_containers!(container, C, names, time_steps, K, meta)

Create optimization containers for all PWMCC concave cut variables and constraints.

Binary and disaggregated variables use sparse (dict-backed) containers keyed by
`(name, k, t)` tuples.  Single-per-`(name, t)` constraints use dense axes
`[names, time_steps]`; per-segment constraints use `[names, 1:K, time_steps]`.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `K::Int`: number of PWMCC sub-intervals
- `meta::String`: meta tag for container keys
"""
function _create_pwmcc_containers!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    K::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    return (
        delta_vars = add_variable_container!(
            container, PiecewiseMcCormickBinary(), C; meta,
        ),
        vd_vars = add_variable_container!(
            container, PiecewiseMcCormickDisaggregated(), C; meta,
        ),
        selector_con = add_constraints_container!(
            container, PiecewiseMcCormickSelectorSum(), C, names, time_steps; meta,
        ),
        linking_con = add_constraints_container!(
            container, PiecewiseMcCormickLinking(), C, names, time_steps; meta,
        ),
        interval_lb_cons = add_constraints_container!(
            container, PiecewiseMcCormickIntervalLB(), C, names, 1:K, time_steps; meta,
        ),
        interval_ub_cons = add_constraints_container!(
            container, PiecewiseMcCormickIntervalUB(), C, names, 1:K, time_steps; meta,
        ),
        chord_ub_con = add_constraints_container!(
            container, PiecewiseMcCormickChordUB(), C, names, time_steps; meta,
        ),
        tangent_lb_l_con = add_constraints_container!(
            container, PiecewiseMcCormickTangentLBL(), C, names, time_steps; meta,
        ),
        tangent_lb_r_con = add_constraints_container!(
            container, PiecewiseMcCormickTangentLBR(), C, names, time_steps; meta,
        ),
    )
end

"""
    _store_pwmcc_result!(containers, name, t, K, result)

Store one `(name, t)` PWMCC result into pre-created containers.

Sparse variable containers are indexed with tuple keys `(name, k, t)`.
Dense constraint containers are indexed with positional axes `[name, t]`
or `[name, k, t]`.
"""
function _store_pwmcc_result!(
    containers::NamedTuple,
    name::String,
    t::Int,
    K::Int,
    result::NamedTuple,
)
    for k in 1:K
        containers.delta_vars[(name, k, t)] = result.delta_vars[k]
        containers.vd_vars[(name, k, t)] = result.vd_vars[k]
        containers.interval_lb_cons[name, k, t] = result.interval_lb_cons[k]
        containers.interval_ub_cons[name, k, t] = result.interval_ub_cons[k]
    end
    containers.selector_con[name, t] = result.selector_con
    containers.linking_con[name, t] = result.linking_con
    containers.chord_ub_con[name, t] = result.chord_ub_con
    containers.tangent_lb_l_con[name, t] = result.tangent_lb_l_con
    containers.tangent_lb_r_con[name, t] = result.tangent_lb_r_con
    return
end
