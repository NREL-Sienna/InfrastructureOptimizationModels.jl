struct NMDTBinaryVariable <: VariableType end
struct NMDTResidualVariable <: VariableType end
struct NMDTBinaryContinuousProductVariable <: VariableType end
struct NMDTResidualProductVariable <: VariableType end

struct NMDTDiscretizationExpression <: ExpressionType end
struct NMDTBinaryContinuousProductExpression <: ExpressionType end
struct NMDTResultExpression <: ExpressionType end

struct NMDTEDiscretizationConstraint <: ConstraintType end
struct NMDTBinaryContinuousProductConstraint <: ConstraintType end
struct NMDTTightenConstraint <: ConstraintType end

struct NormedVariableExpression <: ExpressionType end

struct NMDTDiscretization
    norm_expr
    beta_var
    delta_var
    min::Float64
    max::Float64
    depth::Int
end

function _normed_variable!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    meta::String
) where {C <: IS.InfrastructureSystemsComponent}
    lx = x_max - x_min
    result_expr = add_expression_container!(
        container,
        NormedVariableExpression(),
        C,
        names,
        time_steps;
        meta,
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(-x_min / lx)
        JuMP.add_to_expression!(result, 1.0 / lx, x_var[name, t])
    end
    return result_expr
end

function _discretize!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    beta_var = add_variable_container!(
        container,
        NMDTBinaryVariable(),
        C,
        names,
        1:depth,
        time_steps;
        meta
    )
    delta_var = add_variable_container!(
        container,
        NMDTResidualVariable(),
        C,
        names,
        time_steps;
        meta
    )
    disc_expr = add_expression_container!(
        container,
        NMDTDiscretizationExpression(),
        C,
        names,
        time_steps;
        meta
    )
    disc_cons = add_constraints_container!(
        container,
        NMDTEDiscretizationConstraint(),
        C,
        names,
        time_steps;
        meta
    )

    xh_expr = _normed_variable!(
        container, C, names, time_steps,
        x_var, x_min, x_max, meta
    )

    for name in names, t in time_steps
        disc = disc_expr[name, t] = JuMP.AffExpr(0.0)
        for i in 1:depth
            beta = beta_var[name, i, t] = JuMP.@variable(
                jump_model,
                base_name = "NMDTBinary_$(C)_{$(name), $(t)}",
                binary = true
            )
            JuMP.add_to_expression!(disc, 2.0^(-i), beta)
        end
        delta = delta_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "NMDTResidual_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = 2.0^(-depth)
        )
        JuMP.add_to_expression!(disc, delta)
        disc_cons[name, t] = JuMP.@constraint(
            jump_model,
            xh_expr[name, t] == disc
        )
    end

    return NMDTDiscretization(xh_expr, beta_var, delta_var, x_min, x_max, depth)
end

function _binary_continuous_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    bin_disc,
    cont_var,
    cont_min::Float64,
    cont_max::Float64,
    meta::String;
    tighten::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    depth = bin_disc.depth
    jump_model = get_jump_model(container)

    u_var = add_variable_container!(
        container,
        NMDTBinaryContinuousProductVariable(),
        C,
        names,
        1:depth,
        time_steps;
        meta
    )
    u_cons = add_constraints_container!(
        container,
        NMDTBinaryContinuousProductConstraint(),
        C,
        names,
        1:depth,
        1:4,
        time_steps;
        meta
    )
    result_expr = add_expression_container!(
        container,
        NMDTBinaryContinuousProductExpression(),
        C,
        names,
        time_steps;
        meta
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        for i=1:depth
            u_i = u_var[name, i, t] = JuMP.@variable(
                jump_model,
                base_name = "NMDTBinContProd_$(C)_{$(name), $(i), $(t)}",
                lower_bound = cont_min,
                upper_bound = cont_max
            )
            _add_mccormick_envelope!(
                jump_model, u_cons, (name, i, t),
                cont_var[name, t], bin_disc.beta_var[name, i, t], u_i,
                cont_min, cont_max, 0.0, 1.0;
                lower_bounds = !tighten
            )
            JuMP.add_to_expression!(result, 2.0^(-i), u_i)
        end
    end

    return result_expr
end

function _tighten_lower_bounds!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    result_expr,
    x_disc,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    epigraph_depth = max(2, ceil(Int, 1.5 * x_disc.depth))
    epi_expr = _add_epigraph_quadratic_approx!(
        container, C, names, time_steps,
        x_disc.norm_expr, 0.0, 1.0,
        epigraph_depth, meta * "_epi",
    )
    epi_cons = add_constraints_container!(
        container,
        NMDTTightenConstraint(),
        C,
        names,
        time_steps;
        meta
    )
    for name in names, t in time_steps
        epi_cons[name, t] = JuMP.@constraint(
            jump_model,
            result_expr[name, t] >= epi_expr[name, t],
        )
    end
end

function _residual_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc,
    y_var,
    y_max::Float64,
    meta::String;
    tighten::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    x_max = 2.0^(-x_disc.depth)
    jump_model = get_jump_model(container)

    z_var = add_variable_container!(
        container,
        NMDTResidualProductVariable(),
        C,
        names,
        time_steps;
        meta
    )
    
    for name in names, t in time_steps
        z_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "NMDTResidualProduct_$(C)_{$(name), $(t)}",
            lower_bound = 0.0,
            upper_bound = x_max * y_max,
        )
    end

    _add_mccormick_envelope!(
        container, C, names, time_steps,
        x_disc.delta_var, y_var, z_var,
        0.0, x_max, 0.0, y_max,
        meta; lower_bounds = !tighten
    )

    return z_var
end

function _assemble_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    terms,
    dz_var,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    meta::String;
    result_type = NMDTResultExpression
) where {C <: IS.InfrastructureSystemsComponent}
    x_min, x_max = x_disc.min, x_disc.max
    y_min, y_max = y_disc.min, y_disc.max
    lx = x_max - x_min
    ly = y_max - y_min

    result_expr = add_expression_container!(
        container,
        result_type(),
        C,
        names,
        time_steps;
        meta
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        zh = JuMP.AffExpr(0.0)
        for term in terms
            JuMP.add_to_expression!(zh, term[name, t])
        end
        JuMP.add_to_expression!(zh, dz_var[name, t])
    
        JuMP.add_to_expression!(result, lx * ly, zh)
        JuMP.add_to_expression!(result, lx * y_min, x_disc.norm_expr[name, t])
        JuMP.add_to_expression!(result, ly * x_min, y_disc.norm_expr[name, t])
        JuMP.add_to_expression!(result, x_min * y_min)
    end

    return result_expr
end

function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    meta::String;
    tighten::Bool = false
) where {C <: IS.InfrastructureSystemsComponent}
    bx_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 0.0, 1.0,
        meta * "_bx_xh"
    )
    bx_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc, x_disc.delta_var, 0.0, 2.0^(-x_disc.depth),
        meta * "_bx_dx"
    )

    result_expr = _add_dnmdt_approx!(
        container, C, names, time_steps,
        bx_xh_expr, bx_dx_expr, bx_xh_expr, bx_dx_expr,
        x_disc, x_disc, meta,
        result_type = QuadraticExpression
    )

    if tighten
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, meta
        )
    end

    return result_expr
end

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
    tighten::Bool = false
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta
    )

    return _add_dnmdt_approx!(
        container, C, names, time_steps,
        x_disc, meta; tighten
    )
end

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
        meta
    )
    dz = _residual_product!(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, 1.0, meta
    )

    result_expr = _assemble_product!(
        container, C, names, time_steps,
        [bx_y_expr], dz,
        x_disc, x_disc, meta;
        result_type = QuadraticExpression
    )

    if tighten
        _tighten_lower_bounds!(
            container, C, names, time_steps,
            result_expr, x_disc, meta
        )
    end
end

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
    tighten::Bool = false
) where {C <: IS.InfrastructureSystemsComponent}
    x_disc = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta
    )
    
    return _add_nmdt_approx!(
        container, C, names, time_steps,
        x_disc, meta; tighten
    )
end

_add_nmdt_quadratic_approx! = _add_nmdt_approx!
_add_nmdt_bilinear_approx! = _add_nmdt_approx!
_add_dnmdt_quadratic_approx! = _add_dnmdt_approx!
_add_dnmdt_bilinear_approx! = _add_dnmdt_approx!