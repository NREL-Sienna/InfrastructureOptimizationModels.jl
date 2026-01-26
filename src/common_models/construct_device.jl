"""
Construct device formulation in the optimization container.
This is a two-stage process with ArgumentConstructStage and ModelConstructStage.
"""
function construct_device!(
    container::OptimizationContainer,
    sys::PSY.System,
    ::ArgumentConstructStage,
    model::DeviceModel{D, F},
    network_model::NetworkModel{S},
) where {D <: PSY.Component, F <: AbstractDeviceFormulation, S}
    error(
        "construct_device! not implemented for device type $D with formulation $F " *
        "at ArgumentConstructStage. Implement this method to add variables and expressions.",
    )
end

function construct_device!(
    container::OptimizationContainer,
    sys::PSY.System,
    ::ModelConstructStage,
    model::DeviceModel{D, F},
    network_model::NetworkModel{S},
) where {D <: PSY.Component, F <: AbstractDeviceFormulation, S}
    error(
        "construct_device! not implemented for device type $D with formulation $F " *
        "at ModelConstructStage. Implement this method to add constraints and objectives.",
    )
end

"""
Construct service formulation in the optimization container.
"""
function construct_service!(
    container::OptimizationContainer,
    sys::PSY.System,
    ::ArgumentConstructStage,
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

function construct_service!(
    container::OptimizationContainer,
    sys::PSY.System,
    ::ModelConstructStage,
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

#################################################################################
# Variable and expression multiplier extension points
# These are used by add_to_expression! implementations
#################################################################################

"""
Get the multiplier for a variable type when adding to an expression.
Default implementation returns 1.0. Override for specific variable/device/formulation combinations.
"""
function get_variable_multiplier(
    ::V,
    ::Type{D},
    ::F,
) where {V <: VariableType, D <: PSY.Component, F <: AbstractDeviceFormulation}
    return 1.0
end

function get_variable_multiplier(
    ::V,
    ::D,
    ::F,
) where {V <: VariableType, D <: PSY.Component, F <: AbstractDeviceFormulation}
    return 1.0
end

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
    D <: PSY.Component,
    F <: AbstractDeviceFormulation,
}
    error(
        "get_expression_multiplier not implemented for parameter $P, expression $T, " *
        "device $D, formulation $F. Implement this method in PowerOperationsModels.",
    )
end

"""
Get the multiplier value for a parameter type.
"""
function get_multiplier_value(
    ::P,
    ::D,
    ::F,
) where {P <: ParameterType, D <: PSY.Component, F <: AbstractDeviceFormulation}
    error(
        "get_multiplier_value not implemented for parameter $P, device $D, formulation $F. " *
        "Implement this method in PowerOperationsModels.",
    )
end
