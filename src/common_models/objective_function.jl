"""
Add objective function contributions for devices.
"""
function objective_function!(
    container::OptimizationContainer,
    devices::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
    model::DeviceModel{U, F},
    ::Type{S},
) where {U <: PSY.Component, F <: AbstractDeviceFormulation, S}
    # Default: no objective contribution
    # Override this method to add device-specific objective terms
    return
end
