##################################################
################# SOS Methods ####################
##################################################

# might belong in POM, but here for now.
abstract type VariableValueParameter <: RightHandSideParameter end
"""
Parameter to define unit commitment status updated from the system state
"""
struct OnStatusParameter <: VariableValueParameter end

"""
Parameter to fix a variable value (e.g., from feedforward).
"""
struct FixValueParameter <: VariableValueParameter end

"""
Struct to create the PiecewiseLinearCostConstraint associated with a specified variable.

See [Piecewise linear cost functions](@ref pwl_cost) for more information.
"""
struct PiecewiseLinearCostConstraint <: ConstraintType end

"""
Normalization constraint for PWL cost: sum of delta variables equals on-status.
"""
struct PiecewiseLinearCostNormalizationConstraint <: ConstraintType end

# mildly device-specific, but a single line of code. (Really dispatching on formulation here:
# device is just for safety. Shouldn't have non-thermal with thermal formulation.)
_sos_status(::Type{<:IS.InfrastructureSystemsComponent}, ::AbstractDeviceFormulation) =
    SOSStatusVariable.NO_VARIABLE
_sos_status(::Type{<:PSY.ThermalGen}, ::AbstractThermalUnitCommitment) =
    SOSStatusVariable.VARIABLE

function _get_sos_value(
    container::OptimizationContainer,
    ::Type{V},
    component::T,
) where {T <: IS.InfrastructureSystemsComponent, V <: AbstractDeviceFormulation}
    if skip_proportional_cost(component)
        return SOSStatusVariable.NO_VARIABLE
    elseif has_container_key(container, OnStatusParameter, T)
        return SOSStatusVariable.PARAMETER
    else
        return _sos_status(T, V())
    end
end

_get_sos_value(
    ::OptimizationContainer,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::AbstractServiceFormulation,
) = SOSStatusVariable.NO_VARIABLE

##################################################
################# PWL Variables ##################
##################################################

# This cases bounds the data by 1 - 0
function _add_pwl_variables!(
    container::OptimizationContainer,
    ::Type{T},
    component_name::String,
    time_period::Int,
    cost_data::IS.PiecewiseLinearData,
) where {T <: IS.InfrastructureSystemsComponent}
    var_container = lazy_container_addition!(container, PiecewiseLinearCostVariable(), T)
    # length(PiecewiseStepData) gets number of segments, here we want number of points
    pwlvars = Array{JuMP.VariableRef}(undef, length(cost_data) + 1)
    for i in 1:(length(cost_data) + 1)
        pwlvars[i] =
            var_container[component_name, i, time_period] = JuMP.@variable(
                get_jump_model(container),
                base_name = "PiecewiseLinearCostVariable_$(component_name)_{pwl_$(i), $time_period}",
                lower_bound = 0.0,
                upper_bound = 1.0
            )
    end
    return pwlvars
end

##################################################
################# PWL Constraints ################
##################################################

function _determine_bin_lhs(
    container::OptimizationContainer,
    sos_status::SOSStatusVariable,
    ::Type{T},
    name::String,
    period::Int) where {T <: IS.InfrastructureSystemsComponent}
    if sos_status == SOSStatusVariable.NO_VARIABLE
        return 1.0
        @debug "Using Piecewise Linear cost function but no variable/parameter ref for ON status is passed. Default status will be set to online (1.0)" _group =
            LOG_GROUP_COST_FUNCTIONS
    elseif sos_status == SOSStatusVariable.PARAMETER
        param = OnStatusParameter()
        return get_parameter(container, param, T).parameter_array[name, period]
        @debug "Using Piecewise Linear cost function with parameter OnStatusParameter, $T" _group =
            LOG_GROUP_COST_FUNCTIONS
    elseif sos_status == SOSStatusVariable.VARIABLE
        var = OnVariable()
        return get_variable(container, var, T)[name, period]
        @debug "Using Piecewise Linear cost function with variable OnVariable $T" _group =
            LOG_GROUP_COST_FUNCTIONS
    else
        @assert false
    end
end

# Migration note for POM:
# Old call: _add_pwl_constraint!(container, component, U(), break_points, sos_val, t)
# New call for standard form:
#   power_var = get_variable(container, U(), T)[name, t]
#   _add_pwl_constraint_standard!(container, component, break_points, sos_val, t, power_var)
# New call for compact form (PowerAboveMinimumVariable):
#   power_var = get_variable(container, U(), T)[name, t]
#   P_min = get_active_power_limits(component).min
#   _add_pwl_constraint_compact!(container, component, break_points, sos_val, t, power_var, P_min)

"""
Implement the standard constraints for PWL variables. That is:

```math
\\sum_{k\\in\\mathcal{K}} P_k^{max} \\delta_{k,t} = p_t \\\\
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} = on_t
```

For compact form (PowerAboveMinimumVariable), use `_add_pwl_constraint_compact!` instead.
"""
function _add_pwl_constraint_standard!(
    container::OptimizationContainer,
    component::T,
    break_points::Vector{Float64},
    sos_status::SOSStatusVariable,
    period::Int,
    power_var::JuMP.VariableRef,
    must_run::Bool = false,
) where {T <: IS.InfrastructureSystemsComponent}
    name = get_name(component)
    n_points = length(break_points)

    # Get PWL delta variables
    pwl_var_container = get_variable(container, PiecewiseLinearCostVariable(), T)
    pwl_vars = [pwl_var_container[name, i, period] for i in 1:n_points]

    # Linking constraint: power_var == sum(pwl_vars * breakpoints)
    add_pwl_linking_constraint!(
        container,
        PiecewiseLinearCostConstraint,
        T,
        name,
        period,
        power_var,
        pwl_vars,
        break_points,
    )

    # Normalization constraint: sum(pwl_vars) == on_status
    if must_run
        bin = 1.0
    else
        bin = _determine_bin_lhs(container, sos_status, T, name, period)
    end
    add_pwl_normalization_constraint!(
        container,
        PiecewiseLinearCostNormalizationConstraint,
        T,
        name,
        period,
        pwl_vars,
        bin,
    )
    return
end

"""
Implement the constraints for PWL variables for Compact form. That is:

```math
\\sum_{k\\in\\mathcal{K}} P_k^{max} \\delta_{k,t} = p_t + P_min * u_t \\\\
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} = on_t
```

For standard form, use `_add_pwl_constraint_standard!` instead.
"""
function _add_pwl_constraint_compact!(
    container::OptimizationContainer,
    ::T,
    name::String,
    break_points::Vector{Float64},
    sos_status::SOSStatusVariable,
    period::Int,
    power_var::JuMP.VariableRef,
    P_min::Float64,
    must_run::Bool = false,
) where {T <: IS.InfrastructureSystemsComponent}
    n_points = length(break_points)

    # Get on-status for compact form (needed for both linking and normalization)
    if must_run
        bin = 1.0
    else
        bin = _determine_bin_lhs(container, sos_status, T, name, period)
    end

    # Get PWL delta variables
    pwl_var_container = get_variable(container, PiecewiseLinearCostVariable(), T)
    pwl_vars = [pwl_var_container[name, i, period] for i in 1:n_points]

    # Create constraint container if needed
    if !has_container_key(container, PiecewiseLinearCostConstraint, T)
        con_key = ConstraintKey(PiecewiseLinearCostConstraint, T)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}()
        _assign_container!(
            container.constraints,
            con_key,
            JuMP.Containers.SparseAxisArray(contents),
        )
    end
    con_container = get_constraint(container, PiecewiseLinearCostConstraint(), T)
    jump_model = get_jump_model(container)

    # Compact form linking constraint includes P_min offset
    con_container[name, period] = JuMP.@constraint(
        jump_model,
        bin * P_min + power_var ==
        sum(pwl_vars[i] * break_points[i] for i in 1:n_points)
    )

    # Normalization constraint: sum(pwl_vars) == on_status
    add_pwl_normalization_constraint!(
        container,
        PiecewiseLinearCostNormalizationConstraint,
        T,
        name,
        period,
        pwl_vars,
        bin,
    )
    return
end

##################################################
################ PWL Expressions #################
##################################################

# accepts scaled function data.
function _get_pwl_cost_expression(
    container::OptimizationContainer,
    ::Type{T},
    name::String,
    time_period::Int,
    cost_data::IS.PiecewiseLinearData,
    multiplier::Float64,
) where {T <: IS.InfrastructureSystemsComponent}
    pwl_var_container = get_variable(container, PiecewiseLinearCostVariable(), T)
    gen_cost = JuMP.AffExpr(0.0)
    y_coords_cost_data = IS.get_y_coords(cost_data)
    for (i, cost) in enumerate(y_coords_cost_data)
        JuMP.add_to_expression!(
            gen_cost,
            (cost * multiplier),
            pwl_var_container[name, i, time_period],
        )
    end
    return gen_cost
end

_get_pwl_cost_multiplier(::IS.FuelCurve{IS.PiecewisePointCurve},
    ::VariableType,
    ::AbstractDeviceFormulation,
) = 1.0

_get_pwl_cost_multiplier(::IS.CostCurve{IS.PiecewisePointCurve},
    ::U,
    ::V,
) where {U <: VariableType, V <: AbstractDeviceFormulation} =
    objective_function_multiplier(U(), V())

# FuelCurve/CostCurve: scale for units, then call function data version
function _get_pwl_cost_expression(
    container::OptimizationContainer,
    component::T,
    time_period::Int,
    cost_function::Union{
        IS.FuelCurve{IS.PiecewisePointCurve},
        IS.CostCurve{IS.PiecewisePointCurve},
    },
    ::U,
    ::V,
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    value_curve = IS.get_value_curve(cost_function)
    power_units = IS.get_power_units(cost_function)
    cost_component = IS.get_function_data(value_curve)
    base_power = get_model_base_power(container)
    device_base_power = get_base_power(component)
    cost_data_normalized = get_piecewise_pointcurve_per_system_unit(
        cost_component,
        power_units,
        base_power,
        device_base_power,
    )
    resolution = get_resolution(container)
    dt = Dates.value(resolution) / MILLISECONDS_IN_HOUR
    multiplier = _get_pwl_cost_multiplier(cost_function, U(), V())
    name = get_name(component)
    fuel_consumption_expression = _get_pwl_cost_expression(
        container,
        T,
        name,
        time_period,
        cost_data_normalized,
        dt * multiplier,
    )
    return fuel_consumption_expression
end

##################################################
######## CostCurve: PiecewisePointCurve ##########
##################################################

"""
Add PWL cost terms for data coming from a PiecewisePointCurve
"""
function _add_pwl_term!(
    container::OptimizationContainer,
    component::T,
    cost_function::Union{
        IS.CostCurve{IS.PiecewisePointCurve},
        IS.FuelCurve{IS.PiecewisePointCurve},
    },
    ::U,
    ::V,
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    # multiplier = objective_function_multiplier(U(), V())
    name = get_name(component)
    value_curve = IS.get_value_curve(cost_function)
    cost_component = IS.get_function_data(value_curve)
    base_power = get_model_base_power(container)
    device_base_power = get_base_power(component)
    power_units = IS.get_power_units(cost_function)

    # Normalize data
    data = get_piecewise_pointcurve_per_system_unit(
        cost_component,
        power_units,
        base_power,
        device_base_power,
    )

    if all(iszero.((point -> point.y).(IS.get_points(data))))  # TODO I think this should have been first. before?
        @debug "All cost terms for component $(name) are 0.0" _group =
            LOG_GROUP_COST_FUNCTIONS
        return
    end

    # Compact PWL data does not exists anymore

    cost_is_convex = IS.is_convex(data)
    if !cost_is_convex
        @warn(
            "The cost function provided for $(name) is not compatible with a linear PWL cost function. " *
            "An SOS-2 formulation will be added to the model. This will result in additional binary variables."
        )
    end
    break_points = IS.get_x_coords(data)
    time_steps = get_time_steps(container)
    pwl_cost_expressions = Vector{JuMP.AffExpr}(undef, time_steps[end])
    sos_val = _get_sos_value(container, V, component)
    for t in time_steps
        _add_pwl_variables!(container, T, name, t, data)
        power_var = get_variable(container, U(), T)[name, t]
        _add_pwl_constraint_standard!(
            container,
            component,
            break_points,
            sos_val,
            t,
            power_var,
        )
        if !cost_is_convex
            pwl_var_container = get_variable(container, PiecewiseLinearCostVariable(), T)
            n_points = length(break_points)
            pwl_vars = [pwl_var_container[name, i, t] for i in 1:n_points]
            add_pwl_sos2_constraint!(container, T, name, t, pwl_vars)
        end
        pwl_cost =
            _get_pwl_cost_expression(container, component, t, cost_function, U(), V())
        pwl_cost_expressions[t] = pwl_cost
    end
    return pwl_cost_expressions
end

# ThermalDispatchNoMin implementation is in POM

"""
Creates piecewise linear cost function using a sum of variables and expression with sign and time step included.

# Arguments

  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_function::IS.CostCurve{IS.PiecewisePointCurve}: container for piecewise linear cost
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::C,
    cost_function::IS.CostCurve{IS.PiecewisePointCurve},
    ::U,
) where {
    T <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    component_name = get_name(component)
    @debug "PWL Variable Cost" _group = LOG_GROUP_COST_FUNCTIONS component_name
    # If array is full of tuples with zeros return 0.0
    value_curve = IS.get_value_curve(cost_function)
    cost_component = IS.get_function_data(value_curve)
    if all(iszero.((point -> point.y).(IS.get_points(cost_component))))  # TODO I think this should have been first. before?
        @debug "All cost terms for component $(component_name) are 0.0" _group =
            LOG_GROUP_COST_FUNCTIONS
        return
    end
    pwl_cost_expressions =
        _add_pwl_term!(container, component, cost_function, T(), U())
    for t in get_time_steps(container)
        add_cost_to_expression!(
            container,
            ProductionCostExpression,
            pwl_cost_expressions[t],
            C,
            component_name,
            t,
        )
        add_to_objective_invariant_expression!(container, pwl_cost_expressions[t])
    end
    return
end

"""
Creates piecewise linear cost function using a sum of variables and expression with sign and time step included.
# Arguments
  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_function::IS.FuelCurve{IS.PiecewisePointCurve}: container for piecewise linear cost
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::V,
    cost_function::IS.FuelCurve{IS.PiecewisePointCurve},
    ::U,
) where {
    T <: VariableType,
    V <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    component_name = get_name(component)
    @debug "PWL Variable Cost" _group = LOG_GROUP_COST_FUNCTIONS component_name
    # If array is full of tuples with zeros return 0.0
    value_curve = IS.get_value_curve(cost_function)
    cost_component = IS.get_function_data(value_curve)
    if all(iszero.((point -> point.y).(IS.get_points(cost_component))))  # TODO I think this should have been first. before?
        @debug "All cost terms for component $(component_name) are 0.0" _group =
            LOG_GROUP_COST_FUNCTIONS
        return
    end
    pwl_fuel_consumption_expressions =
        _add_pwl_term!(container, component, cost_function, T(), U())

    # IS getter: simply returns the field of the FuelCurve struct
    is_time_variant_ = is_time_variant(IS.get_fuel_cost(cost_function))
    for t in get_time_steps(container)
        fuel_cost_value = if is_time_variant_
            param = get_parameter_array(container, FuelCostParameter(), V)
            mult = get_parameter_multiplier_array(container, FuelCostParameter(), V)
            param[component_name, t] * mult[component_name, t]
        else
            get_fuel_cost(component)
        end
        pwl_cost_expression = pwl_fuel_consumption_expressions[t] * fuel_cost_value
        add_cost_to_expression!(
            container,
            ProductionCostExpression,
            pwl_cost_expression,
            V,
            component_name,
            t,
        )
        add_cost_to_expression!(
            container,
            FuelConsumptionExpression,
            pwl_fuel_consumption_expressions[t],
            V,
            component_name,
            t,
        )
        if is_time_variant_
            add_to_objective_variant_expression!(container, pwl_cost_expression)
        else
            add_to_objective_invariant_expression!(container, pwl_cost_expression)
        end
    end
    return
end

##################################################
###### CostCurve: PiecewiseIncrementalCurve ######
######### and PiecewiseAverageCurve ##############
##################################################

"""
Creates piecewise linear cost function using a sum of variables and expression with sign and time step included.

# Arguments

  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_function::Union{IS.CostCurve{IS.PiecewiseIncrementalCurve}, IS.CostCurve{IS.PiecewiseAverageCurve}}: container for piecewise linear cost
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::IS.InfrastructureSystemsComponent,
    cost_function::V,
    ::U,
) where {
    T <: VariableType,
    V <: Union{
        IS.CostCurve{IS.PiecewiseIncrementalCurve},
        IS.CostCurve{IS.PiecewiseAverageCurve},
    },
    U <: AbstractDeviceFormulation,
}
    # Create new PiecewisePointCurve
    value_curve = IS.get_value_curve(cost_function)
    power_units = IS.get_power_units(cost_function)
    pointbased_value_curve = IS.InputOutputCurve(value_curve)
    pointbased_cost_function =
        IS.CostCurve(; value_curve = pointbased_value_curve, power_units = power_units)
    # Call method for PiecewisePointCurve
    add_variable_cost_to_objective!(
        container,
        T(),
        component,
        pointbased_cost_function,
        U(),
    )
    return
end

##################################################
###### FuelCurve: PiecewiseIncrementalCurve ######
######### and PiecewiseAverageCurve ##############
##################################################

"""
Creates piecewise linear fuel cost function using a sum of variables and expression with sign and time step included.

# Arguments

  - container::OptimizationContainer : the optimization_container model built in InfrastructureOptimizationModels
  - var_key::VariableKey: The variable name
  - component_name::String: The component_name of the variable container
  - cost_function::Union{IS.FuelCurve{IS.PiecewiseIncrementalCurve}, IS.FuelCurve{IS.PiecewiseAverageCurve}}: container for piecewise linear cost
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::IS.InfrastructureSystemsComponent,
    cost_function::V,
    ::U,
) where {
    T <: VariableType,
    V <: Union{
        IS.FuelCurve{IS.PiecewiseIncrementalCurve},
        IS.FuelCurve{IS.PiecewiseAverageCurve},
    },
    U <: AbstractDeviceFormulation,
}
    # Create new PiecewisePointCurve
    value_curve = IS.get_value_curve(cost_function)
    power_units = IS.get_power_units(cost_function)
    fuel_cost = IS.get_fuel_cost(cost_function)
    pointbased_value_curve = IS.InputOutputCurve(value_curve)
    pointbased_cost_function =
        IS.FuelCurve(;
            value_curve = pointbased_value_curve,
            power_units = power_units,
            fuel_cost = fuel_cost,
        )
    # Call method for PiecewisePointCurve
    add_variable_cost_to_objective!(
        container,
        T(),
        component,
        pointbased_cost_function,
        U(),
    )
    return
end
