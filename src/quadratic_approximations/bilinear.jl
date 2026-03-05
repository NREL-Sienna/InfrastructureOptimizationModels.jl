# Bin2 separable approximation of bilinear products z = x·y.
# Uses the identity: x·y = (1/2)*((x+y)² − x² - y²).
# Calls existing quadratic approximation functions for p²=(x+y)²

struct BilinearApproxSumVariable <: VariableType end              # p = x + y
struct BilinearApproxSumLinkingConstraint <: ConstraintType end   # p == x + y
struct BilinearProductVariable <: VariableType end                # z ≈ x·y

"""
    _add_bilinear_approx_impl!(container, C, names, time_steps, x_var_container, y_var_container, x_min, x_max, y_min, y_max, quad_approx_fn, meta; add_mccormick)

Internal implementation for Bin2 bilinear approximation using z = (1/2)((x+y)² − x² - y²).

Creates auxiliary variables p = x+y, calls `quad_approx_fn` to
approximate p², then combines via multiplicative identity.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var_container`: container of x variables indexed by (name, t)
- `y_var_container`: container of y variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `quad_approx_fn`: callable with signature (container, C, names, ts, var_cont, lo, hi, meta) → Dict
- `meta::String`: identifier for container keys
- `add_mccormick::Bool`: whether to add McCormick envelope constraints (default: false)

# Returns
- `Dict{Tuple{String, Int}, JuMP.AffExpr}`: maps (name, t) to affine expression approximating x·y
"""
function _add_bilinear_approx_impl!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    y_var_container,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    quad_approx_fn,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    # Bounds for p = x + y
    p_min = x_min + y_min
    p_max = x_max + y_max
    IS.@assert_op p_min <= p_max

    jump_model = get_jump_model(container)
    meta_plus = meta * "_plus"
    meta_x = meta * "_x"
    meta_y = meta * "_y"

    # Create p variable container
    p_container = add_variable_container!(
        container,
        BilinearApproxSumVariable(),
        C,
        names,
        time_steps;
        meta = meta_plus,
    )

    # Create linking constraint containers
    p_link_container = add_constraints_container!(
        container,
        BilinearApproxSumLinkingConstraint(),
        C,
        names,
        time_steps;
        meta = meta_plus,
    )

    # Create p variable and linking constraint
    for name in names, t in time_steps
        x = x_var_container[name, t]
        y = y_var_container[name, t]

        p_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "BilinearSum_$(C)_{$(name), $(t)}",
            lower_bound = p_min,
            upper_bound = p_max,
        )

        p_link_container[name, t] =
            JuMP.@constraint(jump_model, p_container[name, t] == x + y)
    end

    # Approximate p² using the provided quadratic approximation function
    zp_dict = quad_approx_fn(
        container,
        C,
        names,
        time_steps,
        p_container,
        p_min,
        p_max,
        meta_plus,
    )
    zx_dict =
        quad_approx_fn(
            container,
            C,
            names,
            time_steps,
            x_var_container,
            x_min,
            x_max,
            meta_x,
        )
    zy_dict =
        quad_approx_fn(
            container,
            C,
            names,
            time_steps,
            y_var_container,
            y_min,
            y_max,
            meta_y,
        )

    # Create z variable container for the bilinear product
    z_container = add_variable_container!(
        container,
        BilinearProductVariable(),
        C,
        names,
        time_steps;
        meta,
    )

    result = Dict{Tuple{String, Int}, JuMP.AffExpr}()

    for name in names, t in time_steps
        z_var = JuMP.@variable(
            jump_model,
            base_name = "BilinearProduct_$(C)_{$(name), $(t)}",
        )
        z_container[name, t] = z_var

        # z = (1/2) * (p² − x² - y²)
        z_expr = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(z_expr, 0.5, zp_dict[(name, t)])
        JuMP.add_to_expression!(z_expr, -0.5, zx_dict[(name, t)])
        JuMP.add_to_expression!(z_expr, -0.5, zy_dict[(name, t)])

        JuMP.@constraint(jump_model, z_var == z_expr)

        result[(name, t)] = JuMP.AffExpr(0.0, z_var => 1.0)
    end

    # Optional McCormick envelope
    if add_mccormick
        _add_mccormick_envelope!(
            container, C, names, time_steps,
            x_var_container, y_var_container, z_container,
            x_min, x_max, y_min, y_max, meta,
        )
    end

    return result
end

"""
    _add_sos2_bilinear_approx!(container, C, names, time_steps, x_var_container, y_var_container, x_min, x_max, y_min, y_max, num_segments, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with solver-native SOS2 quadratic approximations.

# Arguments
Same as `_add_bilinear_approx_impl!` plus:
- `num_segments::Int`: number of PWL segments for each quadratic approximation

# Returns
- `Dict{Tuple{String, Int}, JuMP.AffExpr}`: maps (name, t) to affine expression approximating x·y
"""
function _add_sos2_bilinear_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    y_var_container,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    num_segments::Int,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    quad_fn =
        (cont, CT, nms, ts, vc, lo, hi, m) ->
            _add_sos2_quadratic_approx!(cont, CT, nms, ts, vc, lo, hi, num_segments, m)
    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var_container, y_var_container,
        x_min, x_max, y_min, y_max, quad_fn, meta;
        add_mccormick,
    )
end

"""
    _add_manual_sos2_bilinear_approx!(container, C, names, time_steps, x_var_container, y_var_container, x_min, x_max, y_min, y_max, num_segments, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with manual SOS2 quadratic approximations.

# Arguments
Same as `_add_bilinear_approx_impl!` plus:
- `num_segments::Int`: number of PWL segments for each quadratic approximation

# Returns
- `Dict{Tuple{String, Int}, JuMP.AffExpr}`: maps (name, t) to affine expression approximating x·y
"""
function _add_manual_sos2_bilinear_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    y_var_container,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    num_segments::Int,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    quad_fn =
        (cont, CT, nms, ts, vc, lo, hi, m) ->
            _add_manual_sos2_quadratic_approx!(
                cont,
                CT,
                nms,
                ts,
                vc,
                lo,
                hi,
                num_segments,
                m,
            )
    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var_container, y_var_container,
        x_min, x_max, y_min, y_max, quad_fn, meta;
        add_mccormick,
    )
end

"""
    _add_sawtooth_bilinear_approx!(container, C, names, time_steps, x_var_container, y_var_container, x_min, x_max, y_min, y_max, depth, meta; add_mccormick)

Approximate x·y using Bin2 decomposition with sawtooth quadratic approximations.

# Arguments
Same as `_add_bilinear_approx_impl!` plus:
- `depth::Int`: sawtooth depth (number of binary variables per quadratic approximation)

# Returns
- `Dict{Tuple{String, Int}, JuMP.AffExpr}`: maps (name, t) to affine expression approximating x·y
"""
function _add_sawtooth_bilinear_approx!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var_container,
    y_var_container,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    depth::Int,
    meta::String;
    add_mccormick::Bool = false,
) where {C <: IS.InfrastructureSystemsComponent}
    quad_fn =
        (cont, CT, nms, ts, vc, lo, hi, m) ->
            _add_sawtooth_quadratic_approx!(cont, CT, nms, ts, vc, lo, hi, depth, m)
    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var_container, y_var_container,
        x_min, x_max, y_min, y_max, quad_fn, meta;
        add_mccormick,
    )
end
