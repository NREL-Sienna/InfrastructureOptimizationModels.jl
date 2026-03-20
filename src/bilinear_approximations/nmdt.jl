function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    bx_yh_expr,
    by_dx_expr,
    by_xh_expr,
    bx_dy_expr,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    meta::String;
    lambda::Float64 = DNMDT_LAMBDA,
    result_type::Type = DNMDTResultExpression
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        result_type(),
        C,
        names,
        time_steps;
        meta
    )

    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, y_disc.delta_var, 2.0^(-y_disc.depth),
        meta
    )
    z1_expr = _assemble_product!(
        container, C, names, time_steps,
        [bx_yh_expr, by_dx_expr], dz,
        x_disc, y_disc, meta * "_nmdt1"
    )
    z2_expr = _assemble_product!(
        container, C, names, time_steps,
        [by_xh_expr, bx_dy_expr], dz,
        y_disc, x_disc, meta * "_nmdt2"
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lambda, z1_expr[name, t])
        JuMP.add_to_expression!(result, 1.0 - lambda, z2_expr[name, t])
    end

    return result_expr
end

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
        meta * "_bx_yh"
    )
    by_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        y_disc, x_disc.delta_var, 0.0, 2.0^(-x_disc.depth),
        meta * "_by_dx"
    )
    by_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        y_disc, x_disc.norm_expr, 0.0, 1.0,
        meta * "_by_xh"
    )
    bx_dy_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, y_disc.delta_var, 0.0, 2.0^(-y_disc.depth),
        meta * "_bx_dy"
    )

    return _add_dnmdt_approx!(
        container, C, names, time_steps,
        bx_yh_expr, by_dx_expr, by_xh_expr, bx_dy_expr,
        x_disc, y_disc, meta,
        result_type = BilinearProductExpression
    )
end

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
    meta::String
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta * "_x"
    )
    y_disc = _discretize!(
        container, C, names, time_steps,    
        y_var, y_min, y_max, depth, meta * "_y"
    )

    return _add_dnmdt_approx!(
        container, C, names, time_steps,
        x_disc, y_disc, meta
    )
end

function _add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    yh_expr,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    bx_y_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, yh_expr, 0.0, 1.0,
        meta
    )
    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, yh_expr, 1.0, meta
    )

    return _assemble_product!(
        container, C, names, time_steps,
        [bx_y_expr], dz,
        x_disc, x_disc, meta;
        result_type = BilinearExpression
    )
end

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
        x_var, x_min, x_max, depth, meta
    )
    yh_expr = _normed_variable!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta
    )

    return _add_nmdt_approx!(
        container, C, names, time_steps,
        x_disc, yh_expr, meta
    )        
end