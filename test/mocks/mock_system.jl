"""
Minimal mock for PSY.System without PowerSystems dependency.
Implements only the interface required by OptimizationContainer and models.
"""

mutable struct MockSystem
    base_power::Float64
    components::Dict{DataType, Vector{Any}}
    time_series::Dict{Any, Any}

    MockSystem(base_power = 100.0) = new(base_power, Dict(), Dict())
end

# Required interface methods
get_base_power(sys::MockSystem) = sys.base_power

function get_components(::Type{T}, sys::MockSystem) where {T}
    return get(sys.components, T, T[])
end

function add_component!(sys::MockSystem, component)
    comp_type = typeof(component)
    if !haskey(sys.components, comp_type)
        sys.components[comp_type] = []
    end
    push!(sys.components[comp_type], component)
    return
end

function get_time_series(
    ::Type{T},
    sys::MockSystem,
    component,
    args...;
    kwargs...,
) where {T}
    return get(sys.time_series, (T, component), nothing)
end

function add_time_series!(sys::MockSystem, component, ts)
    sys.time_series[(typeof(ts), component)] = ts
    return
end
