# Generic helpers for adding costs to expressions and objectives
#
# Design principles:
# - Pass (name::String, ::Type{C}) instead of component::C when that's all we need
# - Separate quantity computation from cost conversion
# - Explicit function names (invariant/variant) instead of multiple dispatch
# - No concrete expression or parameter types - caller passes them
# - Single timestep only - looping stays in PSI/POM

#######################################
######## Linear Cost Helpers ##########
#######################################

"""
Add cost term to expression and invariant objective.

Computes `cost = quantity * rate`, adds to target expression (if present),
and adds to the time-invariant part of the objective.

# Arguments
- `container`: the optimization container
- `quantity`: the value being costed (e.g., variable value, expression value)
- `rate`: scalar cost rate (e.g., \$/MWh, \$/MMBTU)
- `E`: target expression type (caller provides, e.g., ProductionCostExpression)
- `C`: component type
- `name`: component name
- `t`: time period
"""
function add_cost_term_invariant!(
    container::OptimizationContainer,
    quantity::JuMPOrFloat,
    rate::Float64,
    ::Type{E},
    ::Type{C},
    name::String,
    t::Int,
) where {E <: ExpressionType, C <: IS.InfrastructureSystemsComponent}
    cost = quantity * rate
    if has_container_key(container, E, C)
        expr = get_expression(container, E(), C)
        JuMP.add_to_expression!(expr[name, t], cost)
    end
    add_to_objective_invariant_expression!(container, cost)
    return cost
end

"""
Add cost term to expression and variant objective.

Fetches rate from parameter, computes `cost = quantity * rate`, adds to target
expression (if present), and adds to the time-variant part of the objective.

# Arguments
- `container`: the optimization container
- `quantity`: the value being costed (e.g., variable value, expression value)
- `P`: parameter type for the rate (caller provides, e.g., FuelCostParameter)
- `E`: target expression type (caller provides, e.g., ProductionCostExpression)
- `C`: component type
- `name`: component name
- `t`: time period
"""
function add_cost_term_variant!(
    container::OptimizationContainer,
    quantity::JuMPOrFloat,
    ::Type{P},
    ::Type{E},
    ::Type{C},
    name::String,
    t::Int,
) where {P <: ParameterType, E <: ExpressionType, C <: IS.InfrastructureSystemsComponent}
    param = get_parameter_array(container, P(), C)
    mult = get_parameter_multiplier_array(container, P(), C)
    rate = param[name, t] * mult[name, t]
    cost = quantity * rate
    if has_container_key(container, E, C)
        expr = get_expression(container, E(), C)
        JuMP.add_to_expression!(expr[name, t], cost)
    end
    add_to_objective_variant_expression!(container, cost)
    return cost
end

#######################################
######## PWL Helper Functions #########
#######################################

"""
Create PWL delta variables for a component at a given time period.

Creates `n_points` variables with specified bounds.

# Arguments
- `container`: the optimization container
- `V`: variable type for the delta variables (caller provides)
- `C`: component type
- `name`: component name
- `t`: time period
- `n_points`: number of PWL points (= number of delta variables)
- `upper_bound`: upper bound for variables (default 1.0 for convex combination formulation;
   use `Inf` for block offer formulation)

# Returns
Vector of the created JuMP variables.
"""
function add_pwl_variables!(
    container::OptimizationContainer,
    ::Type{V},
    ::Type{C},
    name::String,
    t::Int,
    n_points::Int;
    upper_bound::Float64 = 1.0,
) where {V <: SparseVariableType, C <: IS.InfrastructureSystemsComponent}
    # SparseVariableType dispatch automatically creates container with (String, Int, Int) keys
    # axes are (name, pwl_index, time_step).
    var_container = lazy_container_addition!(container, V(), C)
    pwl_vars = Vector{JuMP.VariableRef}(undef, n_points)
    jump_model = get_jump_model(container)
    for i in 1:n_points
        if isfinite(upper_bound)
            pwl_vars[i] =
                var_container[(name, i, t)] = JuMP.@variable(
                    jump_model,
                    base_name = "$(V)_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                    upper_bound = upper_bound,
                )
        else
            pwl_vars[i] =
                var_container[(name, i, t)] = JuMP.@variable(
                    jump_model,
                    base_name = "$(V)_$(C)_{$(name), pwl_$(i), $(t)}",
                    lower_bound = 0.0,
                )
        end
    end
    return pwl_vars
end

"""
Add PWL linking constraint: power variable equals weighted sum of breakpoints.

    P[name, t] == Σ δ[i] * breakpoint[i]

# Arguments
- `container`: the optimization container
- `K`: constraint type (caller provides)
- `C`: component type
- `name`: component name
- `t`: time period
- `power_var`: the power variable to link (JuMP variable reference)
- `pwl_vars`: vector of PWL delta variables
- `breakpoints`: vector of breakpoint values (in p.u.)
"""
function add_pwl_linking_constraint!(
    container::OptimizationContainer,
    ::Type{K},
    ::Type{C},
    name::String,
    t::Int,
    power_var::JuMP.VariableRef,
    pwl_vars::Vector{JuMP.VariableRef},
    breakpoints::Vector{Float64},
) where {K <: ConstraintType, C <: IS.InfrastructureSystemsComponent}
    @assert length(pwl_vars) == length(breakpoints)
    # Create sparse container with (name, time) indexing if it doesn't exist
    if !has_container_key(container, K, C)
        con_key = ConstraintKey(K, C)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}()
        _assign_container!(
            container.constraints,
            con_key,
            JuMP.Containers.SparseAxisArray(contents),
        )
    end
    con_container = get_constraint(container, K(), C)
    jump_model = get_jump_model(container)
    con_container[name, t] = JuMP.@constraint(
        jump_model,
        power_var == sum(pwl_vars[i] * breakpoints[i] for i in eachindex(breakpoints))
    )
    return
end

"""
Add PWL normalization constraint: delta variables sum to on_status.

    Σ δ[i] == on_status

# Arguments
- `container`: the optimization container
- `K`: constraint type (caller provides)
- `C`: component type
- `name`: component name
- `t`: time period
- `pwl_vars`: vector of PWL delta variables
- `on_status`: the on/off status (1.0, or a JuMP variable/parameter)
"""
function add_pwl_normalization_constraint!(
    container::OptimizationContainer,
    ::Type{K},
    ::Type{C},
    name::String,
    t::Int,
    pwl_vars::Vector{JuMP.VariableRef},
    on_status::JuMPOrFloat,
) where {K <: ConstraintType, C <: IS.InfrastructureSystemsComponent}
    # Create sparse container with (name, time) indexing if it doesn't exist
    if !has_container_key(container, K, C)
        con_key = ConstraintKey(K, C)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}()
        _assign_container!(
            container.constraints,
            con_key,
            JuMP.Containers.SparseAxisArray(contents),
        )
    end
    con_container = get_constraint(container, K(), C)
    jump_model = get_jump_model(container)
    con_container[name, t] = JuMP.@constraint(
        jump_model,
        sum(pwl_vars) == on_status
    )
    return
end

"""
Add SOS2 constraint for PWL variables (required for non-convex curves).

# Arguments
- `container`: the optimization container
- `C`: component type
- `name`: component name
- `t`: time period
- `pwl_vars`: vector of PWL delta variables
"""
function add_pwl_sos2_constraint!(
    container::OptimizationContainer,
    ::Type{C},
    name::String,
    t::Int,
    pwl_vars::Vector{JuMP.VariableRef},
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)
    n_points = length(pwl_vars)
    JuMP.@constraint(jump_model, pwl_vars in MOI.SOS2(collect(1:n_points)))
    return
end

"""
Compute PWL cost expression from delta variables and slopes.

Returns the cost expression without adding it to the objective (caller decides
whether to use invariant or variant).

    cost = Σ δ[i] * slope[i] * multiplier

# Arguments
- `pwl_vars`: vector of PWL delta variables
- `slopes`: vector of slope values (cost per segment, already normalized)
- `multiplier`: additional multiplier (e.g., dt for time resolution)

# Returns
JuMP affine expression representing the cost.
"""
function get_pwl_cost_expression(
    pwl_vars::Vector{JuMP.VariableRef},
    slopes::Vector{Float64},
    multiplier::Float64,
)
    @assert length(pwl_vars) == length(slopes)
    cost = JuMP.AffExpr(0.0)
    for (i, slope) in enumerate(slopes)
        JuMP.add_to_expression!(cost, slope * multiplier, pwl_vars[i])
    end
    return cost
end
