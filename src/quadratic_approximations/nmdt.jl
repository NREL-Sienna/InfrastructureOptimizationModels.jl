# NMDT (Normalized Multiparametric Disaggregation Technique) quadratic approximation of x².
# Normalizes x to [0,1], discretizes using L binary variables β₁,…,β_L plus a
# residual δ ∈ [0, 2^{−L}], then replaces each binary-continuous product β_i·xh
# with a McCormick-linearized auxiliary variable. Assembles the result via the
# separable identity x² = (lx·xh + x_min)². Optionally tightens with an epigraph
# lower bound on xh².
# NMDT Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation
# technique for global optimization of polynomial programming problems.

"""
    _add_dnmdt_approx!(container, C, names, time_steps, x_disc, meta; tighten)

Approximate x² using the Double NMDT (DNMDT) method from a pre-built discretization.

Constructs two binary-continuous products (β·xh and β·δ) and delegates to the core
DNMDT assembler, storing results in a `QuadraticExpression` container. Optionally
tightens lower bounds with an epigraph relaxation via `_tighten_lower_bounds!`.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `meta::String`: identifier encoding the original variable type being approximated
- `tighten::Bool`: if true, add epigraph lower-bound tightening (default: false)
"""
function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    meta::String;
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    bx_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 0.0, 1.0,
        meta * "_bx_xh"; tighten,
    )
    bx_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.delta_var, 0.0, 2.0^(-x_disc.depth),
        meta * "_bx_dx"; tighten,
    )

    result_expr = _add_dnmdt_approx!(
        container, C, names, time_steps,
        bx_xh_expr, bx_dx_expr, bx_xh_expr, bx_dx_expr,
        x_disc, x_disc, meta; tighten,
        result_type = QuadraticExpression,
    )

    if tighten
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, meta,
        )
    end

    return result_expr
end

"""
    _add_dnmdt_approx!(container, C, names, time_steps, x_var, x_min, x_max, depth, meta; tighten)

Approximate x² using the Double NMDT (DNMDT) method from raw variable inputs.

Discretizes x via `_discretize!` then delegates to the `NMDTDiscretization` overload.
Stores results in a `QuadraticExpression` container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `depth::Int`: number of binary discretization levels L
- `meta::String`: identifier encoding the original variable type being approximated
- `tighten::Bool`: if true, add epigraph lower-bound tightening (default: false)
"""
function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta,
    )

    return _add_dnmdt_approx!(
        container, C, names, time_steps,
        x_disc, meta; tighten,
    )
end

"""
    _add_nmdt_approx!(container, C, names, time_steps, x_disc, meta; tighten)

Approximate x² using the NMDT method from a pre-built discretization.

Computes the binary-continuous product β·xh and residual product δ·xh, then
assembles x² via `_assemble_product!`. Stores results in a `QuadraticExpression`
container. Optionally tightens lower bounds with an epigraph relaxation.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `meta::String`: identifier encoding the original variable type being approximated
- `tighten::Bool`: if true, add epigraph lower-bound tightening (default: false)
"""
function _add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    meta::String;
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    bx_y_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 0.0, 1.0,
        meta; tighten,
    )
    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 1.0, meta;
        tighten,
    )

    result_expr = _assemble_product!(
        container, C, names, time_steps,
        [bx_y_expr], dz,
        x_disc, x_disc, meta;
        result_type = QuadraticExpression,
    )

    if tighten
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, meta,
        )
    end

    return result_expr
end

"""
    _add_nmdt_approx!(container, C, names, time_steps, x_var, x_min, x_max, depth, meta; tighten)

Approximate x² using the NMDT method from raw variable inputs.

Discretizes x via `_discretize!` then delegates to the `NMDTDiscretization` overload.
Stores results in a `QuadraticExpression` container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of variables indexed by (name, t)
- `x_min::Float64`: lower bound of x domain
- `x_max::Float64`: upper bound of x domain
- `depth::Int`: number of binary discretization levels L
- `meta::String`: identifier encoding the original variable type being approximated
- `tighten::Bool`: if true, add epigraph lower-bound tightening (default: false)
"""
function _add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta,
    )

    return _add_nmdt_approx!(
        container, C, names, time_steps,
        x_disc, meta; tighten,
    )
end

# Aliases used by the quadratic and bilinear approximation dispatch layers.
# Both quadratic (x²) and bilinear (x·y with x=y) cases share the same implementation.
_add_nmdt_quadratic_approx! = _add_nmdt_approx!
_add_dnmdt_quadratic_approx! = _add_dnmdt_approx!
