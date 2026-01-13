"""
Extension point: Is the variable binary/integer?
"""
function get_variable_binary(
    ::T,
    ::Type{U},
    ::F,
) where {T <: VariableType, U <: PSY.Component, F <: AbstractDeviceFormulation}
    return false
end

"""
Extension point: Get variable lower bound.
"""
function get_variable_lower_bound(
    ::T,
    d::U,
    ::F,
) where {T <: VariableType, U <: PSY.Component, F <: AbstractDeviceFormulation}
    return nothing
end

"""
Extension point: Get variable upper bound.
"""
function get_variable_upper_bound(
    ::T,
    d::U,
    ::F,
) where {T <: VariableType, U <: PSY.Component, F <: AbstractDeviceFormulation}
    return nothing
end

"""
Extension point: Get variable warm start value.
"""
function get_variable_warm_start_value(
    ::T,
    d::U,
    ::F,
) where {T <: VariableType, U <: PSY.Component, F <: AbstractDeviceFormulation}
    return nothing
end

# Service variable overloads
function get_variable_binary(
    ::T,
    ::Type{U},
    ::F,
) where {T <: VariableType, U <: PSY.Service, F <: AbstractServiceFormulation}
    return false
end

function get_variable_lower_bound(
    ::T,
    service::U,
    d::V,
    ::F,
) where {T <: VariableType, U <: PSY.Service, V <: PSY.Component, F <: AbstractServiceFormulation}
    return nothing
end

function get_variable_upper_bound(
    ::T,
    service::U,
    d::V,
    ::F,
) where {T <: VariableType, U <: PSY.Service, V <: PSY.Component, F <: AbstractServiceFormulation}
    return nothing
end

function get_variable_warm_start_value(
    ::T,
    d::U,
    ::F,
) where {T <: VariableType, U <: PSY.Component, F <: AbstractServiceFormulation}
    return nothing
end
