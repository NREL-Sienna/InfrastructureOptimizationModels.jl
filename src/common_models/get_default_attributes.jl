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
