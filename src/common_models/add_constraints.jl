"""
Add constraints to the optimization container.
"""
function add_constraints!(
    container::OptimizationContainer,
    ::Type{T},
    devices::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
    model::DeviceModel{U, F},
    network_model::NetworkModel{S},
) where {T <: ConstraintType, U <: PSY.Component, F <: AbstractDeviceFormulation, S}
    error(
        "add_constraints! not implemented for constraint type $T, " *
        "device type $U with formulation $F. Implement this method to add constraints.",
    )
end
