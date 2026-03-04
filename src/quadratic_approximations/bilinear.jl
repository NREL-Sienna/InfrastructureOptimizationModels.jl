# Bin2 separable approximation of bilinear products z = x·y.
# Uses the difference-of-squares identity: x·y = ¼((x+y)² − (x−y)²).
# Calls existing quadratic approximation functions twice for u² and v².

struct BilinearApproxSumVariable <: VariableType end              # u = x + y
struct BilinearApproxDiffVariable <: VariableType end             # v = x − y
struct BilinearApproxSumLinkingConstraint <: ConstraintType end   # u == x + y
struct BilinearApproxDiffLinkingConstraint <: ConstraintType end  # v == x − y
struct BilinearProductVariable <: VariableType end                # z ≈ x·y

"""
    _add_bilinear_approx_impl!(container, C, names, time_steps, x_var_container, y_var_container, x_min, x_max, y_min, y_max, quad_approx_fn, meta; add_mccormick)

Internal implementation for Bin2 bilinear approximation using z = ¼((x+y)² − (x−y)²).

Creates auxiliary variables u = x+y and v = x−y, calls `quad_approx_fn` twice to
approximate u² and v², then combines via the difference-of-squares identity.

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
    # Bounds for u = x + y and v = x − y
    u_min = x_min + y_min
    u_max = x_max + y_max
    v_min = x_min - y_max
    v_max = x_max - y_min
    IS.@assert_op u_max > u_min
    IS.@assert_op v_max > v_min

    jump_model = get_jump_model(container)
    meta_plus = meta * "_plus"
    meta_minus = meta * "_minus"

    # Create u and v variable containers
    u_container = add_variable_container!(
        container,
        BilinearApproxSumVariable(),
        C,
        names,
        time_steps;
        meta = meta_plus,
    )
    v_container = add_variable_container!(
        container,
        BilinearApproxDiffVariable(),
        C,
        names,
        time_steps;
        meta = meta_minus,
    )

    # Create linking constraint containers
    u_link_container = add_constraints_container!(
        container,
        BilinearApproxSumLinkingConstraint(),
        C,
        names,
        time_steps;
        meta = meta_plus,
    )
    v_link_container = add_constraints_container!(
        container,
        BilinearApproxDiffLinkingConstraint(),
        C,
        names,
        time_steps;
        meta = meta_minus,
    )

    # Create u, v variables and linking constraints
    for name in names, t in time_steps
        x = x_var_container[name, t]
        y = y_var_container[name, t]

        u_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "BilinearSum_$(C)_{$(name), $(t)}",
            lower_bound = u_min,
            upper_bound = u_max,
        )
        v_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "BilinearDiff_$(C)_{$(name), $(t)}",
            lower_bound = v_min,
            upper_bound = v_max,
        )

        u_link_container[name, t] =
            JuMP.@constraint(jump_model, u_container[name, t] == x + y)
        v_link_container[name, t] =
            JuMP.@constraint(jump_model, v_container[name, t] == x - y)
    end

    # Approximate u² and v² using the provided quadratic approximation function
    zu_dict = quad_approx_fn(container, C, names, time_steps, u_container, u_min, u_max, meta_plus)
    zv_dict = quad_approx_fn(container, C, names, time_steps, v_container, v_min, v_max, meta_minus)

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

        # z = 0.25 * (u² − v²)
        z_expr = JuMP.AffExpr(0.0)
        JuMP.add_to_expression!(z_expr, 0.25, zu_dict[(name, t)])
        JuMP.add_to_expression!(z_expr, -0.25, zv_dict[(name, t)])

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
    quad_fn = (cont, CT, nms, ts, vc, lo, hi, m) ->
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
    quad_fn = (cont, CT, nms, ts, vc, lo, hi, m) ->
        _add_manual_sos2_quadratic_approx!(cont, CT, nms, ts, vc, lo, hi, num_segments, m)
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
    quad_fn = (cont, CT, nms, ts, vc, lo, hi, m) ->
        _add_sawtooth_quadratic_approx!(cont, CT, nms, ts, vc, lo, hi, depth, m)
    return _add_bilinear_approx_impl!(
        container, C, names, time_steps,
        x_var_container, y_var_container,
        x_min, x_max, y_min, y_max, quad_fn, meta;
        add_mccormick,
    )
end
