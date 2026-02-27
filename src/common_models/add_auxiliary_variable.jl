@doc raw"""
Default implementation of adding auxiliary variable to the model.
"""
function add_variables!(
    container::OptimizationContainer,
    ::Type{T},
    devices::U,
    formulation,
) where {
    T <: AuxVariableType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
} where {D <: IS.InfrastructureSystemsComponent}
    @assert !isempty(devices)
    var_type = T()
    time_steps = get_time_steps(container)
    add_aux_variable_container!(
        container,
        var_type,
        D,
        PSY.get_name.(devices),
        time_steps,
    )
    return
end
