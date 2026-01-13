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
