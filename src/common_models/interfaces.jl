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
    ::SYSTEM_TYPE,
    ::M,
    model::DeviceModel{D, F},
    network_model::NetworkModel{S},
) where {M <: ConstructStage, D <: COMP_TYPE, F <: AbstractDeviceFormulation, S}
    error(
        "construct_device! not implemented for device type $D with formulation $F " *
        "at $(_to_string(M)). Implement this method to add variables and expressions.",
    )
end

# for some reason this one doesn't print the stage name.
function construct_service!(
    ::OptimizationContainer,
    ::SYSTEM_TYPE,
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
) where {U <: COMP_TYPE, F <: AbstractDeviceFormulation, S}
    # Default: no objective contribution
    # Override this method to add device-specific objective terms
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
) where {T <: ConstraintType, U <: COMP_TYPE, F <: AbstractDeviceFormulation, S}
    error(
        "add_constraints! not implemented for constraint type $T, " *
        "device type $U with formulation $F. Implement this method to add constraints.",
    )
end

"""
Extension point: Add parameters to the optimization container.
Concrete implementations are in PowerOperationsModels.
"""
function add_parameters! end

"""
Add variable cost to the objective function. Stub implementation.
"""
function add_variable_cost!(
    ::OptimizationContainer,
    ::T,
    devices::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
    ::F,
) where {T <: VariableType, U <: COMP_TYPE, F <: AbstractDeviceFormulation}
    # Default: no cost
    # Override this method to add device-specific variable costs
    return
end

###############################
###### get_foo functions ######
###############################

# Variable multipliers: default to 1.0

"""
Get the multiplier for a variable type when adding to an expression.
Default implementation returns 1.0. Override for specific variable/device/formulation combinations.
"""
function get_variable_multiplier(
    ::V,
    ::Type{D},
    ::F,
) where {V <: VariableType, D <: COMP_TYPE, F <: AbstractDeviceFormulation}
    return 1.0
end

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
    D <: COMP_TYPE,
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
    d::U,
    ::F,
) where {T <: TimeSeriesParameter, U <: COMP_TYPE, F <: AbstractDeviceFormulation}
    return 1.0  # Default: no scaling
end

"""
Get the multiplier value for a parameter type.
"""
function get_multiplier_value(
    ::P,
    ::D,
    ::F,
) where {P <: ParameterType, D <: COMP_TYPE, F <: AbstractDeviceFormulation}
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
) where {U <: COMP_TYPE, F <: AbstractDeviceFormulation}
    return Dict{String, Any}()
end

"""
Extension point: Get default time series names for a device formulation.
"""
function get_default_time_series_names(
    ::Type{U},
    ::Type{F},
) where {U <: COMP_TYPE, F <: AbstractDeviceFormulation}
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
    U <: COMP_TYPE,
    F <: Union{AbstractDeviceFormulation, AbstractServiceFormulation},
}
    return false
end

"""
Extension point: Get variable lower bound.
"""
function get_variable_lower_bound(
    ::T,
    d::U,
    ::F,
) where {
    T <: VariableType,
    U <: COMP_TYPE,
    F <: Union{AbstractDeviceFormulation, AbstractServiceFormulation},
}
    return nothing
end

"""
Extension point: Get variable upper bound.
"""
function get_variable_upper_bound(
    ::T,
    d::U,
    ::F,
) where {
    T <: VariableType,
    U <: COMP_TYPE,
    F <: Union{AbstractDeviceFormulation, AbstractServiceFormulation},
}
    return nothing
end

"""
Extension point: Get variable warm start value.
"""
function get_variable_warm_start_value(
    ::T,
    d::U,
    ::F,
) where {
    T <: VariableType,
    U <: COMP_TYPE,
    F <: Union{AbstractDeviceFormulation, AbstractServiceFormulation},
}
    return nothing
end

###############################
###### Proportional Cost ######
###############################

"""
Extension point: Get proportional cost term from operation cost data.
Non-time-varying signature - returns a single cost value for all time steps.
"""
function proportional_cost(
    op_cost,
    ::VariableType,
    d::COMP_TYPE,
    ::AbstractDeviceFormulation,
)
    return 0.0
end

"""
Extension point: Get proportional cost term from operation cost data.
Time-varying signature - may return different values per time step.
"""
function proportional_cost(
    ::OptimizationContainer,
    op_cost,
    ::VariableType,
    d::IS.InfrastructureSystemsComponent,
    ::AbstractDeviceFormulation,
    ::Int,
)
    return 0.0
end

"""
Extension point: Check if proportional cost term is time-variant.
Returns true if the cost should be added to the variant objective expression.
"""
function is_time_variant_term(
    ::OptimizationContainer,
    op_cost,
    ::VariableType,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::AbstractDeviceFormulation,
    ::Int,
)
    return false
end

# stub so we can have operation cost in mock components
get_operation_cost(::IS.InfrastructureSystemsComponent) = nothing

# stub for must_run check - device-specific implementations in POM
get_must_run(::IS.InfrastructureSystemsComponent) = false

sos_status(::Any, ::AbstractServiceFormulation) = SOSStatusVariable.NO_VARIABLE

###############################
###### Start-up Cost ##########
###############################

"""
Extension point: Convert raw startup cost to a scalar value.
Device-specific implementations (e.g., for StartUpStages, MultiStartVariable) are in POM.
"""
start_up_cost(
    cost,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::VariableType,
    ::AbstractDeviceFormulation,
) = cost

###############################
###### Build-pipeline ext #####
###############################

"""
Extension point: Construct all services for a given build stage.
Called from `build_impl!`. Concrete implementations in PowerOperationsModels.
"""
function construct_services! end

"""
Extension point: Construct the network model for a given build stage.
Called from `build_impl!`. Concrete implementations in PowerOperationsModels.
"""
function construct_network! end

"""
Extension point: Construct the HVDC network model.
Called from `build_impl!`. Concrete implementations in PowerOperationsModels.
"""
function construct_hvdc_network! end

"""
Extension point: Add power flow evaluation data to the container.
Default: no-op (handles the common case of no power flow evaluators).
"""
function add_power_flow_data!(
    ::OptimizationContainer,
    ::Vector{<:AbstractPowerFlowEvaluationModel},
    ::PSY.System,
)
    return
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
Extension point: Write optimization results to the store.
Concrete implementations in PSI (simulation layer).
"""
function write_results! end

"""
Extension point: variable cost.
"""
function variable_cost(
    cost::PSY.OperationalCost,
    ::V,
    ::Type{C},
    ::F,
) where {
    V <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    F <: AbstractDeviceFormulation,
}
    error(
        "Unimplemented function: couldn't handle operation cost of type $(typeof(cost)) " *
        " on $C component with variable $V and formulation $F.",
    )
end

variable_cost(
    ::Nothing,
    ::VariableType,
    ::Type{<:IS.InfrastructureSystemsComponent},
    ::AbstractDeviceFormulation,
) = 0.0

function get_initial_conditions_device_model(
    model::OperationModel,
    device_model::DeviceModel,
)
    error(
        "get_initial_conditions_device_model not implemented for model type $(typeof(model)) and device model type $(typeof(device_model)). Implement this method in PowerOperationsModels.",
    )
end

"""
Extension point: get the initial condition type for a given constraint, device, and formulation.
Concrete implementations in POM.
"""
function _get_initial_condition_type(X, Y, Z)
    error("`_get_initial_condition_type` must be implemented for $X, $Y and $Z")
end
