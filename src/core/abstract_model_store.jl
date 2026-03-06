# Store const definitions
# Update src/simulation/simulation_store_common.jl with any changes.
const STORE_CONTAINER_DUALS = :duals
const STORE_CONTAINER_PARAMETERS = :parameters
const STORE_CONTAINER_VARIABLES = :variables
const STORE_CONTAINER_AUX_VARIABLES = :aux_variables
const STORE_CONTAINER_EXPRESSIONS = :expressions
const STORE_CONTAINERS = (
    STORE_CONTAINER_DUALS,
    STORE_CONTAINER_PARAMETERS,
    STORE_CONTAINER_VARIABLES,
    STORE_CONTAINER_AUX_VARIABLES,
    STORE_CONTAINER_EXPRESSIONS,
)
const STORE_CONTAINER_TYPES = (
    ConstraintType,
    ParameterType,
    VariableType,
    AuxVariableType,
    ExpressionType,
)

# Derives from store_field_for_type in optimization_container_utils.jl
get_store_container_type(
    ::OptimizationContainerKey{T, U},
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType} = store_field_for_type(T)

abstract type AbstractModelStore end

# Required fields for subtypes
# - :duals
# - :parameters
# - :variables
# - :aux_variables
# - :expressions

# Required methods for subtypes:
# - read_optimizer_stats
#
# Each subtype must have a field for each instance of STORE_CONTAINERS.

function Base.empty!(store::T) where {T <: AbstractModelStore}
    for (name, type) in zip(fieldnames(T), fieldtypes(T))
        val = get_data_field(store, name)
        try
            empty!(val)
        catch
            @error "Base.empty! must be customized for type $T or skipped"
            rethrow()
        end
    end
end

# FIXME getproperty allows for more customization than getfield, but we're not actually
# using that flexibility anywhere downstream...
# PERF symbols likely resolve at runtime. Would be better to do type and @generated.
get_data_field(store::AbstractModelStore, type::Symbol) = getproperty(store, type)

function Base.isempty(store::T) where {T <: AbstractModelStore}
    for (name, type) in zip(fieldnames(T), fieldtypes(T))
        val = get_data_field(store, name)
        try
            !isempty(val) && return false
        catch
            @error "Base.isempty must be customized for type $T or skipped"
            rethrow()
        end
    end

    return true
end

@generated function list_fields(
    store::AbstractModelStore,
    ::Type{T},
) where {T <: OptimizationKeyType}
    field = QuoteNode(store_field_for_type(T))
    return :(return keys(getfield(store, $field)))
end

@generated function list_keys(
    store::AbstractModelStore,
    ::Type{T},
) where {T <: OptimizationKeyType}
    field = QuoteNode(store_field_for_type(T))
    return :(return collect(keys(getfield(store, $field))))
end

@generated function get_value(
    store::AbstractModelStore,
    ::T,
    ::Type{U},
) where {T <: OptimizationKeyType, U <: InfrastructureSystemsType}
    K = key_for_type(T)
    field = QuoteNode(store_field_for_type(T))
    return :(return getfield(store, $field)[$K(T, U)])
end
