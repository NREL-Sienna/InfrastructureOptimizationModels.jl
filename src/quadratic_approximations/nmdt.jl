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

struct NMDTDiscretization
    norm_expr
    beta_var
    delta_var
    min::Float64
    max::Float64
    depth::Int
end

function _assemble_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    bc_prods,
    dz_var,
    x_disc::NMDTDiscretization,
    y_disc::NMDTDIscretization,
    meta::String,
)
    x_min, x_max = x_disc.min, x_disc.max
    y_min, y_max = y_disc.min, y_disc.max
    lx = x_max - x_min
    ly = y_max - y_min

    result_expr = add_expression_container!(
        container,
        NMDTResultExpression(),
        C,
        names,
        time_steps;
        meta
    )

    for name in names, t in time_steps
        result = result[name, t] = JuMP.AffExpr(0.0)
        zh = JuMP.AffExpr(0.0)
        for term in terms
            JuMP.add_to_expression!(zh, term)
        end
        JuMP.add_to_expression!(zh, dz_var)
    
        JuMP.add_to_expression!(result, lx * ly, zh)
        JuMP.add_to_expression!(result, lx * y_min, xh_expr[name, t])
        JuMP.add_to_expression!(result, ly * x_min, yh_expr[name, t])
        JuMP.add_to_expression!(result, x_min * y_min)
    end

    return result_expr
end

function _add_dmndt_approx!(
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
    meta::String,
)
    dz = _residual_product(

    )
    z1_expr = _assemble_product!(
        container, C, names, time_steps
        [bx_yh_expr, by_dx_expr], dz,
        x_disc, y_disc, meta
    )
    z2_expr = _assemble_product!(
        [by_xh_expr, bx_dy_expr],
        y_disc, x_disc, meta
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lambda, z1_expr)
        JuMP.add_to_expression!(result, 1.0 - lambda, z2_expr)
    end

    return result
end

function _add_dmndt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    y_disc::NMDTDIscretization,
    meta::String,
)
    bx_yh_expr = _binary_continuous_product!(
        x_disc.beta, y_disc.norm_expr, 0.0, 1.0
    )
    by_dx_expr = _binary_continuous_product!(
        y_disc.beta, x_disc.delta, 0.0, 2.0^(-x_disc.depth)
    )
    by_xh_expr = _binary_continuous_product!(
        y_disc.beta, x_disc.norm_expr, 0.0, 1.0
    )
    bx_dy_expr = _binary_continuous_product!(
        x_disc.beta, y_disc.delta, 0.0, 2.0^(-y_disc.depth)
    )

    return _add_dmndt_approx!(
        container, C, names, time_steps,
        bx_yh_expr, by_dx_expr, by_xh_expr, bx_dy_expr,
        x_disc, y_disc, meta
    )
end

function _add_dmndt_approx!(
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
    meta::String
)
    x_disc = _discretize(
        container, C, names, time_steps,
        x_var, x_min, x_max, meta * "_x"
    )
    y_disc = _discretize(
        container, C, names, time_steps,    
        y_var, y_min, y_max, meta * "_y"
    )

    return _add_dmndt_approx!(
        container, C, names, time_steps,
        x_disc, y_disc, meta
    )
end

function _add_dmndt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    meta::String,
)
    bx_xh_expr = _binary_continuous_product!(
        x_disc.beta, x_disc.norm_expr, 0.0, 1.0
    )
    bx_dx_expr = _binary_continuous_product!(
        x_disc.beta, x_disc.delta, 0.0, 2.0^(-x_disc.depth)
    )

    return _add_dmndt_approx!(
        container, C, names, time_steps,
        bx_xh_expr, bx_dx_expr, bx_xh_expr, bx_dx_expr,
        x_disc, x_disc, meta
    )
end

function _add_dmndt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    meta::String,
)
    x_disc = _discretize(
        container, C, names, time_steps
        x_var, x_min, x_max, meta
    )

    return _add_dmndt_approx!(
        container, C, names, time_steps,
        x_disc, meta
    )
end

function add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    y_var,
    y_min::Float64,
    y_max::Float64,
    meta::String,
)
    bx_y_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        x_disc.beta, yh_expr, 0.0, 1.0,
        meta
    )
    dz = _residual_product!(
        x_disc, yh_expr
    )
    return _assemble_product!(
        [bx_y_expr],
        dz
    )
end

function add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    x_min::Float64,,
    x_max::Float64,,
    y_min::Float64,,
    y_max::Float64,
    meta::String,
)
    x_disc = _discretize(
        container, C, names, time_steps,
        x_var, x_min, x_max, meta
    )

    return add_nmdt_approx!(
        container, C, names, time_steps,
        x_disc, y_var, y_min, y_max, meta
    )        
end

function add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    x_min::Float64,
    x_max::Float64,
    meta::String,
)
    return add_nmdt_approx!(
        container, C, names, time_steps,
        x_var, x_var, x_min, x_max, x_min, x_max,
        meta
    )
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
)
    lx = x_max - x_min
    result_expr = add_expression_container!(
        container,
        DNMDTScaledVariableExpression(),
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
    xh_expr,
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
    bin_var,
    cont_var,
    cont_min::Float64,
    cont_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
    scale::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
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
        for i=1:x_disc.depth
            u_i = u_var[name, i, t] = JuMP.@variable(
                jump_model,
                base_name = "NMDTBinContProd_$(C)_{$(name), $(i), $(t)}",
                lower_bound = cont_min,
                upper_bound = cont_max
            )
            _add_mccormick_envelope!(
                jump_model, u_cons, (name, i, t),
                cont_var[name, t], bin_var[name, i, t], u_i,
                cont_min, cont_max, 0.0, 1.0;
                lower_bounds = !tighten
            )
            JuMP.add_to_expression!(result, 2.0^(-i), u_i)
        end
    end

    return result_expr
end

function _tighten!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    z_expr,
    x_var,
    x_min::Float64,
    x_max::Float64,
    epigraph_depth::Int,
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    epi_expr = _add_epigraph_quadratic_approx!(
        container, C, names, time_steps,
        x_var, x_min, x_max,
        epigraph_depth, meta * "_epi",
    )
    epi_cons = add_constraints_container!(
        container,
        NMDTTightenConstraint(),
        C,
        names,
        time_steps;
        meta = meta * "_epi_lb",
    )
    for name in names, t in time_steps
        epi_cons[name, t] = JuMP.@constraint(
            jump_model,
            z_expr[name, t] >= epi_expr[name, t],
        )
    end
end

function _residual_product!(
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
    meta::String;
    tighten::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    z_var = add_variable_container!(
        container,
        NMDTResidualProductVariable(),
        C,
        names,
        time_steps;
        meta
    )
    
    corners = (x_min * y_min, x_min * y_max, x_max * y_min, x_max * y_max)
    for name in names, t in time_steps
        dz_var[name, t] = JuMP.@variable(
            jump_model,
            base_name = "NMDTResidualProduct_$(C)_{$(name), $(t)}",
            lower_bound = min(corners),
            upper_bound = max(corners),
        )
    end

    _add_mccormick_envelope!(
        container, C, names, time_steps,
        x_var, y_var, z_var,
        x_min, x_max, y_min, y_max,
        meta; lower_bounds = !tighten
    )

    return dz_var
end

function _residual_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    y_disc::NMDTDiscretization,
    meta::String;
    tighten::Bool = true
)
    return _residual_product!(
        container, C, names, time_steps,
        x_disc.delta, y_disc.delta,
        x_disc.dmin, x_disc.dmax,
        y_disc.dmin, y_disc.dmax,
        meta; tighten
    )
end

function _residual_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    yh_expr,
    meta::String;
    tighten::Bool = true
)
    return _residual_product!(
        container, C, names, time_steps,
        x_disc.delta, yh_expr,
        x_disc.dmin, x_disc.dmax,
        0.0, 1.0,
        meta; tighten
    )
end

function _discretized_continuous_product!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    bx_yh_expr,
    yh_expr,
    dx_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    result_type::Type = NMDTResultExpression,
    tighten::Bool = false
) where {C <: IS.InfrastructureSystemsComponent}
    lx = x_max - x_min

    result_expr = add_expression_container!(
        container,
        result_type(),
        C,
        names,
        time_steps;
        meta
    )

    # $\Delta_z=\Delta_xy$
    dz_var = _residual_product(
        container, C, names, time_steps,
        dx_var, y_var, y_min, y_max,
        depth, meta; tighten
    )

    for name in names, t in time_steps
        # $\hat{z}=\sum_{i=1}\beta_i^xy+\Delta_z
        zh = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(zh, bx_y_expr[name, t])
        JuMP.add_to_expression!(zh, dz_var[name, t])

        # $z=xy=(l_x\hat{x}+\underline{x})y=l_x\hat{z}+\underline{x}y$
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lx, bx_y_expr[name, t])
        JuMP.add_to_expression!(result, x_min, y_var[name, t])
    end

    return result_expr
end

# Compute the product between two discretized variables, where the individual
# binary * continuous and continuous * continuous expressions have been
# predetermined.
function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    xh_expr,
    bx_yh_expr,
    by_dx_expr,
    yh_expr,
    by_xh_expr,
    bx_dy_expr,
    dx_var,
    dy_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
    result_type::Type = NMDTResultExpression,
    lambda::Float64 = DNMDT_LAMBDA
) where {C <: IS.InfrastructureSystemsComponent}
    result_expr = add_expression_container!(
        container,
        DNMDTProductExpression(),
        C,
        names,
        time_steps;
        meta
    )

    dnmdt1 = _assemble_product(
        container, C, names, time_steps,
        [bx_yh_expr, by_dx_expr],
        xh_expr, yh_expr, dz_var,
        x_min, x_max, y_min, y_max,
        meta * "_nmdt1"
    )
    dnmdt2 = _assemble_product!(
        container, C, names, time_steps,
        [by_xh_expr, bx_dy_expr],
        yh_expr, xh_expr, dz_var,
        y_min, y_max, x_min, x_max,
        meta * "_nmdt2"
    )

    for name in names, t in time_steps
        result = result_expr[name, t] = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(result, lambda, dnmdt1[name t])
        JuMP.add_to_expression!(result, 1.0 - lambda, dnmdt2[name, t])
    end

    return result_expr
end


function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    xh_expr,
    beta_x_var,
    dx_var,
    yh_expr,
    beta_y_var,
    dy_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    add_mccormick::Bool = false
) where {C <: IS.InfrastructureSystemsComponent}
    eps_L = 2.0^(-depth)
    bx_yh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_x_var, yh_expr, y_min, y_max,
        depth, meta * "_bx_y"
    )
    by_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_y_var, dx_var, 0.0, eps_L,
        depth, meta * "_by_dx"
    )
    by_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_y_var, xh_expr, x_min, x_max,
        depth, meta * "_by_x"
    )
    bx_dy_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_x_var, dy_var, 0.0, eps_L,
        depth, meta * "_bx_dy"
    )

    result_expr = _add_dnmdt_approx!(
        container, C, names, time_steps,
        xh_expr, bx_yh_expr, by_dx_expr,
        yh_expr, by_xh_expr, bx_dy_expr,
        dz_var, x_min, x_max, y_min, y_max,
        depth, meta;
    )

    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var, y_var, result_expr,
            x_min, x_max, y_min, y_max,
            meta
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
    y_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    add_mccormick::Bool = false
) where {C <: IS.InfrastructureSystemsComponent}
    meta_x = meta * "_x"
    meta_y = meta * "_y"

    xh_expr = _normed_variable!(
        container, C, names, time_steps,
        x_var, x_min, x_max, meta_x
    )
    yh_expr = _normed_variable!(
        container, C, names, time_steps,
        y_var, y_min, y_max, meta_y
    )

    beta_x_var, dx_var = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta_x
    )
    beta_y_var, dy_var = _discretize!(
        container, C, names, time_steps,
        y_var, y_min, y_max, depth, meta_y
    )

    return _add_dnmdt_approx!(
        container, C, names, time_steps,
        xh_expr, beta_x_var, dx_var,
        yh_expr, beta_y_var, dy_var,
        x_min, x_max, y_min, y_max,
        depth, meta
    )
end

function _add_dnmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    beta_x_var,
    dx_var,
    xh_expr,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth))
) where {C <: IS.InfrastructureSystemsComponent}
    eps_L = 2.0^(-depth)
    bx_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_x_var, xh_expr, x_min, x_max,
        depth, meta * "_bx_x"; tighten
    )
    bx_dx_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_x_var, dx_var, 0.0, eps_L,
        depth, meta * "_bx_dx"; tighten
    )

    result_expr = _add_dnmdt_approx!(
        container, C, names, time_steps,
        bx_xh_expr, bx_dx_expr, bx_xh_expr, bx_dx_expr,
        dx_var, dx_var, xh_expr, xh_expr,
        x_min, x_max, x_min, x_max,
        depth, meta; tighten, result_type = QuadraticExpression
    )
    if tighten
        _tighten!(
            container, C, names, time_steps,
            result_expr, x_var, x_min, x_max,
            epigraph_depth, meta
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
    tighten::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth))
) where {C <: IS.InfrastructureSystemsComponent}
    xh_expr = _normed_variable!(
        container, C, names, time_steps,
        x_var, x_min, x_max, meta

    )
    beta_x_var, dx_var = _discretize!(
        container, C, names, time_steps,
        xh_expr, depth, meta
    )

    return _add_dnmdt_quadratic_approx!(
        container, C, names, time_steps,
        beta_x_var, dx_var, xh_expr,
        x_min, x_max, depth, meta;
        tighten, epigraph_depth
    )
end

# Approximate the product between two continuous variables by discretizing
# one variable and solving the simpler binary * continuous products.
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
    meta::String;
) where {C <: IS.InfrastructureSystemsComponent}
    beta_x_var, dx_var = _discretize!(
        container, C, names, time_steps,
        x_var, x_min, x_max, depth, meta
    )

    bx_y_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_x_var, y_var, y_min, y_max,
        depth, meta
    )

    return _discretized_continuous_product!(
        container, C, names, time_steps,
        bx_y_expr, y_var, dx_var,
        x_min, x_max, y_min, y_max,
        depth, meta;
        result_type = BilinearProductExpression
    )
end

function _add_nmdt_approx!(

)   
    container, C, names, time_steps,
    [bx_xh_expr], xh_expr, xh_expr, dz_var,
    x_min, x_max, x_min, x_max,
    meta
)
end

function _add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    xh_expr,
    beta_x_var,
    dx_var,
    x_min::Float64,
    x_max::Float64,
    depth::Int,
    meta::String;
    tighten::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth))
)
    bx_xh_expr = _binary_continuous_product!(
        container, C, names, time_steps,
        beta_x_var, xh_expr, x_min, x_max,
        depth, meta; tighten
    )

    return _assemble_product!(
        container, C, names, time_steps,
        [bx_xh_expr], xh_expr, xh_expr, dz_var,
        x_min, x_max, x_min, x_max,
        meta
    )
end

function _add_nmdt_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_disc::NMDTDiscretization,
    meta
)
    bx_xh_expr = _binary_continuous_product(
        container, C, names, time_steps,
        x_disc, x_disc.norm_expr, meta
    )
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
    tighten::Bool = false,
    epigraph_depth::Int = max(2, ceil(Int, 1.5 * depth))
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