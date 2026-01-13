"""
Add time series parameters to the optimization container.
"""
function add_parameters!(
    container::OptimizationContainer,
    ::Type{T},
    devices::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
    model::DeviceModel{U, F},
) where {T <: TimeSeriesParameter, U <: PSY.Component, F <: AbstractDeviceFormulation}
    # Default stub implementation
    # Override this for specific parameter types that need custom behavior
    # Most time series parameters will use the standard pattern
    @debug "add_parameters! called for $T, $U with formulation $F"
    return
end

"""
Extension point: Get default time series name for a parameter type.
Returns the name of the time series to look for in the component.
"""
function get_time_series_name(
    ::T,
    d::U,
    model::DeviceModel{U, F},
) where {T <: TimeSeriesParameter, U <: PSY.Component, F <: AbstractDeviceFormulation}
    # Check if model has time series names configured
    ts_names = get_time_series_names(model)
    if haskey(ts_names, T)
        return ts_names[T]
    end

    # Default: use parameter type name without "TimeSeriesParameter" suffix
    param_name = string(T)
    return replace(param_name, "TimeSeriesParameter" => "")
end

"""
Extension point: Get multiplier value for a time series parameter.
This scales the time series values for each device.
"""
function get_multiplier_value(
    ::T,
    d::U,
    ::F,
) where {T <: TimeSeriesParameter, U <: PSY.Component, F <: AbstractDeviceFormulation}
    return 1.0  # Default: no scaling
end
