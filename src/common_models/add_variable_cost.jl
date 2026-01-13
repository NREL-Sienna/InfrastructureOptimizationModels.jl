"""
Add variable cost to the objective function.
"""
function add_variable_cost!(
    container::OptimizationContainer,
    ::T,
    devices::Union{Vector{U}, IS.FlattenIteratorWrapper{U}},
    ::F,
) where {T <: VariableType, U <: PSY.Component, F <: AbstractDeviceFormulation}
    # Default: no cost
    # Override this method to add device-specific variable costs
    return
end
