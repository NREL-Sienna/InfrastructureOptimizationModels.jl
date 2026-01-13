"""
Extension point: Get default attributes for a device formulation.
"""
function get_default_attributes(
    ::Type{U},
    ::Type{F},
) where {U <: PSY.Component, F <: AbstractDeviceFormulation}
    return Dict{String, Any}()
end
