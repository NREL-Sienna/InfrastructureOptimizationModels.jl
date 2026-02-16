##########################################
###### make_system_expressions! ##########
##########################################

"""
Extension point: Create system-level balance expressions for a network formulation.
Concrete implementations are in PowerOperationsModels for each network model type.
"""
function make_system_expressions! end

###############################
###### construct_device! ######
###############################

_to_string(::Type{ArgumentConstructStage}) = "ArgumentConstructStage"
_to_string(::Type{ModelConstructStage}) = "ModelConstructStage"

"""
Stub implementation: downstream modules should implement `construct_device!` for 
`ArgumentConstructStage` and for `ModelConstructStage` stages separately.
"""
function construct_device!(
    ::OptimizationContainer,
    ::IS.ComponentContainer,
    ::M,
    model::DeviceModel{D, F},
    network_model::NetworkModel{S},
) where {
    M <: ConstructStage,
    D <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
    S,
}
    error(
        "construct_device! not implemented for device type $D with formulation $F " *
        "at $(_to_string(M)). Implement this method to add variables and expressions.",
    )
end

# for some reason this one doesn't print the stage name.
function construct_service!(
    ::OptimizationContainer,
    ::IS.ComponentContainer,
    ::ConstructStage,
    model::ServiceModel{S, F},
    devices_template::Dict{Symbol, DeviceModel},
    incompatible_device_types::Set{<:DataType},
    network_model::NetworkModel{N},
) where {S <: PSY.Service, F <: AbstractServiceFormulation, N}
    error(
        "construct_service! not implemented for service type $S with formulation $F. " *
        "Implement this method in PowerOperationsModels.",
    )
end

###############################
###### add_foo functions ######
###############################

# previously called objective_function!, but renamed to be more consistent with others.
"""
Add objective function contributions for devices.
"""
function add_to_objective_function!(
    ::OptimizationContainer,
    ::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
    ::DeviceModel{U, F},
    ::Type{S},
) where {
    U <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
    S <: AbstractPowerModel,
}
    error(
        "add_to_objective_function! not implemented for device type $U with formulation $F and power model $S.",
    )
    return
end

"""
Add constraints to the optimization container. Stub implementation.
"""
function add_constraints!(
    ::OptimizationContainer,
    ::Type{T},
    devices::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
    model::DeviceModel{U, F},
    network_model::NetworkModel{S},
) where {
    T <: ConstraintType,
    U <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
    S,
}
    error(
        "add_constraints! not implemented for constraint type $T, " *
        "device type $U with formulation $F. Implement this method to add constraints.",
    )
end

"""
Extension point: Add parameters to the optimization container.
Concrete implementations are in PowerOperationsModels.
"""
function add_parameters!(
    ::OptimizationContainer,
    ::Type{T},
    devices::U,
    model::DeviceModel{D, W},
) where {
    T <: ParameterType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    W <: AbstractDeviceFormulation,
} where {D <: IS.InfrastructureSystemsComponent}
    error(
        "add_parameters! not implemented for parameter type $T, device type $D with formulation $W. Implement this method in PowerOperationsModels.",
    )
end

###############################
###### get_foo functions ######
###############################

# Variable multipliers: default to 1.0

"""
Get the multiplier for a variable type when adding to an expression.
Default implementation returns 1.0. Override for specific variable/device/formulation combinations.
"""
get_variable_multiplier(
    ::VariableType,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::AbstractDeviceFormulation,
) = 1.0

# Expression multipliers: error by default.
"""
Get the multiplier for an expression type based on parameter type.
"""
function get_expression_multiplier(
    ::P,
    ::Type{T},
    ::D,
    ::F,
) where {
    P <: ParameterType,
    T <: ExpressionType,
    D <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    error(
        "get_expression_multiplier not implemented for parameter $P, expression $T, " *
        "device $D, formulation $F. Implement this method in PowerOperationsModels.",
    )
end

# parameter multipliers: time series defaults to 1.0, other types error by default.

"""
Extension point: Get multiplier value for a time series parameter.
This scales the time series values for each device.
"""
function get_multiplier_value(
    ::T,
    ::U,
    ::F,
) where {
    T <: TimeSeriesParameter,
    U <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    return 1.0  # Default: no scaling
end

"""
Get the multiplier value for a parameter type.
"""
function get_multiplier_value(
    ::P,
    ::D,
    ::F,
) where {
    P <: ParameterType,
    D <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    error(
        "get_multiplier_value not implemented for parameter $P, device $D, formulation $F. " *
        "Implement this method in PowerOperationsModels.",
    )
end

# stuff associated to a formulation: attributes, time series names
"""
Extension point: Get default attributes for a device formulation.
"""
function get_default_attributes(
    ::Type{U},
    ::Type{F},
) where {U <: IS.InfrastructureSystemsComponent, F <: AbstractDeviceFormulation}
    return Dict{String, Any}()
end

"""
Extension point: Get default time series names for a device formulation.
"""
function get_default_time_series_names(
    ::Type{U},
    ::Type{F},
) where {U <: IS.InfrastructureSystemsComponent, F <: AbstractDeviceFormulation}
    return Dict{Type{<:ParameterType}, String}()
end

# variable properties, for device or service formulation
"""
Extension point: Is the variable binary/integer?
"""
function get_variable_binary(
    ::T,
    ::Type{U},
    ::F,
) where {
    T <: VariableType,
    U <: IS.InfrastructureSystemsComponent,
    F <: Union{AbstractDeviceFormulation, AbstractServiceFormulation},
}
    error("`get_variable_binary` not implemented for $T and $U (with formulation $F).")
end

"""
Extension point: Get variable lower bound.
"""
get_variable_lower_bound(
    ::VariableType,
    ::IS.InfrastructureSystemsComponent,
    ::Union{AbstractDeviceFormulation, AbstractServiceFormulation},
) = nothing

"""
Extension point: Get variable upper bound.
"""
get_variable_upper_bound(
    ::VariableType,
    ::IS.InfrastructureSystemsComponent,
    ::Union{AbstractDeviceFormulation, AbstractServiceFormulation},
) = nothing

"""
Extension point: Get variable warm start value.
"""
get_variable_warm_start_value(
    ::VariableType,
    ::IS.InfrastructureSystemsComponent,
    ::Union{AbstractDeviceFormulation, AbstractServiceFormulation},
) = nothing

###############################
###### Proportional Cost ######
###############################

"""
Extension point: Get proportional cost term from operation cost data.
Non-time-varying signature - returns a single cost value for all time steps.
"""
function proportional_cost(
    ::O,
    ::V,
    ::C,
    ::F,
) where {
    O <: PSY.OperationalCost,
    V <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    error(
        "proportional cost not implemented for non-time-varying case for cost type $O, variable type $V, component type $C, formulation $F.",
    )
end

"""
Extension point: Get proportional cost term from operation cost data.
Time-varying signature - may return different values per time step.
"""
function proportional_cost(
    ::OptimizationContainer,
    ::O,
    ::V,
    ::C,
    ::F,
    ::Int,
) where {
    O <: PSY.OperationalCost,
    V <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    error(
        "proportional cost not implemented for time-varying case for cost type $O, variable type $V, component type $C, formulation $F.",
    )
end

"""
Extension point: Check if proportional cost term is time-variant.
Returns true if the cost should be added to the variant objective expression.
"""
is_time_variant_term(
    ::OptimizationContainer,
    ::PSY.OperationalCost,
    ::VariableType,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::AbstractDeviceFormulation,
    ::Int,
) = false

# corresponds to get_must_run for thermals, but avoiding device specific code here.
"""
Extension point: whether to skip adding proportional cost for a given device.

For thermals, equivalent to `get_must_run`, but that implementation belongs in POM.
"""
skip_proportional_cost(d::IS.InfrastructureSystemsComponent) = false

###############################
###### Start-up Cost ##########
###############################

"""
Extension point: Convert raw startup cost to a scalar value.
Device-specific implementations (e.g., for StartUpStages, MultiStartVariable) are in POM.
"""
function start_up_cost(
    cost::Any, # could be NamedTuple, StartUpStages, AffExpr, or Float. 
    ::Type{T},
    ::V,
    ::F,
) where {
    T <: IS.InfrastructureSystemsComponent,
    V <: VariableType,
    F <: AbstractDeviceFormulation,
}
    error(
        "start_up_cost not implemented for cost type $(typeof(cost)), device type $T, " *
        "variable type $V, formulation $F.",
    )
end

###############################
###### Build-pipeline ext #####
###############################

"""
Extension point: Construct all services for a given build stage.
Called from `build_impl!`. Concrete implementations in PowerOperationsModels.
"""
function construct_services!(
    ::OptimizationContainer,
    ::IS.ComponentContainer,
    ::ConstructStage,
    ::ServicesModelContainer,
    ::DevicesModelContainer,
    ::NetworkModel{T},
) where {T <: AbstractPowerModel}
    error("construct_services! not implemented for network model with power model $T.")
end

"""
Extension point: Construct the network model for a given build stage.
Called from `build_impl!`. Concrete implementations in PowerOperationsModels.
"""
function construct_network!(
    ::OptimizationContainer,
    ::NetworkModel{T},
    ::ProblemTemplate,
) where {T <: AbstractPowerModel}
    error("construct_network! not implemented for network model with power model $T.")
end

"""
Extension point: Construct the HVDC network model.
Called from `build_impl!`. Concrete implementations in PowerOperationsModels.
"""
function construct_hvdc_network!(
    ::OptimizationContainer,
    ::IS.ComponentContainer,
    ::NetworkModel{T},
    ::H,
    ::ProblemTemplate,
) where {T <: AbstractPowerModel, H <: AbstractHVDCNetworkModel}
    error(
        "construct_hvdc_network! not implemented for network model with power model $T and HVDC model $H.",
    )
end

"""
Extension point: Add power flow evaluation data to the container.
Default: no-op (handles the common case of no power flow evaluators).
"""
function add_power_flow_data!(
    ::OptimizationContainer,
    evaluators::Vector{<:AbstractPowerFlowEvaluationModel},
    ::IS.ComponentContainer,
)
    if !isempty(evaluators)
        error(
            "Power flow in-the-loop with the new IOM-POM-PSI split isn't working yet.",
        )
    end
end

"""
Extension point: Solve the power flow model.
Default: error. Concrete implementations require PowerFlows integration.
"""
function solve_powerflow! end

"""
Extension point: Calculate auxiliary variable values.
Concrete implementations in PowerOperationsModels for specific aux variable types.
"""
function calculate_aux_variable_value! end

"""
Extension point: Check if an auxiliary variable type comes from power flow evaluation.
Default: false. Override in POM for PowerFlowAuxVariableType subtypes.
"""
is_from_power_flow(::Type{<:AuxVariableType}) = false

"""
Extension point: Get minimum and maximum limits for a given component, constraint type, and device formulation.
"""
get_min_max_limits(
    ::IS.InfrastructureSystemsComponent,
    ::Type{<:ConstraintType},
    ::AbstractDeviceFormulation,
) = nothing

"""
Extension point: variable cost.

The one exception where it isn't just `get_variable(cost)`: storage devices, where we 
need to map `ActivePower{In/Out}` to {charge/discharge} variable cost.
"""
function variable_cost(
    cost::PSY.OperationalCost,
    ::VariableType,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::AbstractDeviceFormulation,
)
    return PSY.get_variable(cost)
end

variable_cost(
    ::Nothing,
    ::VariableType,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::AbstractDeviceFormulation,
) = 0.0

get_initial_conditions_device_model(
    ::OperationModel,
    model::DeviceModel{T, FixedOutput},
) where {T <: PSY.Device} = model

"""
Extension point: get the initial condition type for a given constraint, device, and formulation.
Concrete implementations in POM. Used for ramp constraints.
"""
_get_initial_condition_type(
    X::Type{<:ConstraintType},
    Y::Type{<:PSY.Component},
    Z::Type{<:AbstractDeviceFormulation},
) = error("`_get_initial_condition_type` not implemented for $X , $Y and $Z")
