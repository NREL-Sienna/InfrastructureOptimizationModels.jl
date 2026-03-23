# DNMDT (Double Normalized Multiparametric Disaggregation Technique) bilinear approximation of x·y.
# Independently discretizes both x and y, forms four cross binary-continuous products, then
# combines two NMDT estimates with a convex weighting λ (default 0.5). Reduces to the NMDT
# formulation when applied to x·x (quadratic case).
# Reference: Teles, Castro, Matos (2013), Multiparametric disaggregation technique for global
# optimization of polynomial programming problems.

"""
    _add_dnmdt_approx!(container, C, names, time_steps, x_disc, y_disc, meta)

Approximate x·y using the DNMDT method from pre-built discretizations.

Constructs all four cross binary-continuous products (β_x·yh, β_y·δx, β_y·xh, β_x·δy)
then delegates to the core DNMDT assembler. Stores results in a `BilinearProductExpression`
container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `y_disc::NMDTDiscretization`: pre-built discretization for y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    bx_yh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, y_disc.norm_expr, 0.0, 1.0,
        meta * "_bx_yh",
    )
    by_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        y_disc, x_disc.delta_var, 0.0, 2.0^(-x_disc.depth),
        meta * "_by_dx",
    )
    by_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        y_disc, x_disc.norm_expr, 0.0, 1.0,
        meta * "_by_xh",
    )
    bx_dy_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, y_disc.delta_var, 0.0, 2.0^(-y_disc.depth),
        meta * "_bx_dy",
    )

    return _add_dnmdt_approx!(
        container, C, names, time_steps,
        bx_yh_expr, by_dx_expr, by_xh_expr, bx_dy_expr,
        x_disc, y_disc, meta;
        result_type = BilinearProductExpression,
    )
end

"""
    _add_dnmdt_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta)

Approximate x·y using the DNMDT method from raw variable inputs.

Discretizes both x and y independently via `_discretize!` then delegates to the
two-discretization overload. Stores results in a `BilinearProductExpression` container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `depth::Int`: number of binary discretization levels L for both x and y
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta * "_x",
    )
    y_disc = _discretize!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta * "_y",
    )

    return _add_dnmdt_approx!(
        container, C, names, time_steps,
        x_disc, y_disc, meta,
    )
end

"""
    _add_nmdt_approx!(container, C, names, time_steps, x_disc, yh_expr, meta)

Approximate x·y using the NMDT method from a pre-built x discretization and normalized y.

Discretizes only x (using `x_disc`) while y is already normalized to yh ∈ [0,1].
Computes binary-continuous product β_x·yh and residual product δ_x·yh, then assembles
x·y via `_assemble_product!`. Stores results in a `BilinearProductExpression` container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_disc::NMDTDiscretization`: pre-built discretization for x
- `yh_expr`: expression container for the normalized variable yh = (y − y_min)/(y_max − y_min)
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    yh_expr,
    y_min::Float64,
    y_max::Float64,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    bx_y_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, yh_expr, 0.0, 1.0,
        meta,
    )
    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, yh_expr, 1.0, meta;
    )

    return _assemble_product!(
        container, C, names, time_steps,
        [bx_y_expr], dz,
        x_disc, yh_expr, y_min, y_max, meta;
        result_type = BilinearProductExpression,
    )
end

"""
    _add_nmdt_approx!(container, C, names, time_steps, x_var, y_var, x_min, x_max, y_min, y_max, depth, meta)

Approximate x·y using the NMDT method from raw variable inputs.

Discretizes x via `_discretize!` and normalizes y via `_normed_variable!`, then
delegates to the `(x_disc, yh_expr)` overload. Stores results in a `BilinearProductExpression`
container.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `depth::Int`: number of binary discretization levels L for x
- `meta::String`: identifier encoding the original variable type being approximated
"""
function _add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta * "_x",
    )
    yh_expr = _normed_variable!(
        container, C, names, time_steps,
        y_var, y_min, y_max, meta * "_y"
    )

    return _add_nmdt_approx!(
        container, C, names, time_steps,
        x_disc, yh_expr, y_min, y_max, meta,
    )
end

_add_nmdt_bilinear_approx! = _add_nmdt_approx!
_add_dnmdt_bilinear_approx! = _add_dnmdt_approx!