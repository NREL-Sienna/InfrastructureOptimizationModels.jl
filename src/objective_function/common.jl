# there's also a ReserveDemandCurve version in POM.
"""
Add a cost expression term to a cost-related expression container.
"""
function add_cost_to_expression!(
    container::OptimizationContainer,
    ::Type{S},
    cost_expression::JuMPOrFloat,
    component::T,
    time_period::Int,
) where {
    S <: Union{CostExpressions, FuelConsumptionExpression},
    T <: IS.InfrastructureSystemsComponent,
}
    if has_container_key(container, S, T)
        device_cost_expression = get_expression(container, S(), T)
        component_name = PSY.get_name(component)
        JuMP.add_to_expression!(
            device_cost_expression[component_name, time_period],
            cost_expression,
        )
    end
    return
end

# TODO export this.
# set to -1.0 for loads in POM
objective_function_multiplier(::VariableType, ::AbstractDeviceFormulation) = 1.0

##################################
#### ActivePowerVariable Cost ####
##################################

function add_variable_cost!(
    container::OptimizationContainer,
    ::U,
    devices::IS.FlattenIteratorWrapper{T},
    ::V,
) where {T <: PSY.Component, U <: VariableType, V <: AbstractDeviceFormulation}
    for d in devices
        op_cost_data = PSY.get_operation_cost(d)
        add_variable_cost_to_objective!(container, U(), d, op_cost_data, V())
        _add_vom_cost_to_objective!(container, U(), d, op_cost_data, V())
    end
    return
end

##################################
########## VOM Cost ##############
##################################

# called in market bid cost and above in ActivePowerVariable cost.
function _add_vom_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::PSY.Component,
    op_cost::PSY.OperationalCost,
    ::U,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    variable_cost_data = variable_cost(op_cost, T(), component, U())
    power_units = PSY.get_power_units(variable_cost_data)
    vom_cost = PSY.get_vom_cost(variable_cost_data)
    multiplier = 1.0 # VOM Cost is always positive
    cost_term = PSY.get_proportional_term(vom_cost)
    iszero(cost_term) && return
    base_power = get_model_base_power(container)
    device_base_power = PSY.get_base_power(component)
    cost_term_normalized = get_proportional_cost_per_system_unit(
        cost_term,
        power_units,
        base_power,
        device_base_power,
    )
    resolution = get_resolution(container)
    dt = Dates.value(resolution) / MILLISECONDS_IN_HOUR

    name = PSY.get_name(component)
    C = typeof(component)
    rate = cost_term_normalized * multiplier * dt

    for t in get_time_steps(container)
        variable = get_variable(container, T(), C)[name, t]
        add_cost_term_invariant!(
            container,
            variable,
            rate,
            ProductionCostExpression,
            C,
            name,
            t,
        )
    end
    return
end

# FIXME move, thin wrapper around add_variable_cost_to_objective!.
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::PSY.Component,
    op_cost::PSY.OperationalCost,
    ::U,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    variable_cost_data = variable_cost(op_cost, T(), component, U())
    add_variable_cost_to_objective!(container, T(), component, variable_cost_data, U())
    return
end

# FIXME not actually called anywhere?

#=
function _get_cost_function_parameter_container(
    container::OptimizationContainer,
    ::S,
    component::T,
    ::U,
    ::V,
    cost_type::Type{W},
) where {
    S <: ObjectiveFunctionParameter,
    T <: PSY.Component,
    U <: VariableType,
    V <: Union{AbstractDeviceFormulation, AbstractServiceFormulation},
    W,
}
    if has_container_key(container, S, T)
        return get_parameter(container, S(), T)
    else
        container_axes = axes(get_variable(container, U(), T))
        if has_container_key(container, OnStatusParameter, T)
            sos_val = SOSStatusVariable.PARAMETER
        else
            sos_val = sos_status(component, V())
        end
        return add_param_container!(
            container,
            S(),
            T,
            U,
            sos_val,
            uses_compact_power(component, V()),
            W,
            container_axes...,
        )
    end
end
=#

##################################################
################## Fuel Cost #####################
##################################################

"""
Parameter to define fuel cost time series
"""
struct FuelCostParameter <: ObjectiveFunctionParameter end

# used in quadratic_curve and piecewise_linear objective functions.
function _add_time_varying_fuel_variable_cost!(
    container::OptimizationContainer,
    ::T,
    component::V,
    fuel_cost::IS.TimeSeriesKey,
) where {T <: VariableType, V <: PSY.Component}
    expression = get_expression(container, FuelConsumptionExpression(), V)
    name = PSY.get_name(component)
    for t in get_time_steps(container)
        add_cost_term_variant!(
            container,
            expression[name, t],
            FuelCostParameter,
            ProductionCostExpression,
            V,
            name,
            t,
        )
    end
    return
end

# Used for dispatch (on/off decision) for devices where operation_cost::Union{MarketBidCost, FooCost}
# currently: ThermalGen, ControllableLoad subtypes.

# FIXME only called in POM, device specific code.
function _onvar_cost(::PSY.CostCurve{PSY.PiecewisePointCurve})
    # OnVariableCost is included in the Point itself for PiecewisePointCurve
    return 0.0
end

function _onvar_cost(
    cost_function::Union{PSY.CostCurve{IS.LinearCurve}, PSY.CostCurve{IS.QuadraticCurve}},
)
    value_curve = PSY.get_value_curve(cost_function)
    cost_component = PSY.get_function_data(value_curve)
    # Always in \$/h
    constant_term = PSY.get_constant_term(cost_component)
    return constant_term
end

function _onvar_cost(::PSY.CostCurve{PSY.PiecewiseIncrementalCurve})
    # Input at min is used to transform to InputOutputCurve
    return 0.0
end

function _onvar_cost(
    ::OptimizationContainer,
    cost_function::PSY.CostCurve{T},
    ::PSY.Component,
    ::Int,
) where {T <: IS.ValueCurve}
    return _onvar_cost(cost_function)
end
