# note that we skip duals because they correspond to constraint keys.
field_for_key(::Type{<:VariableKey}) = :variables
field_for_key(::Type{<:AuxVarKey}) = :aux_variables
field_for_key(::Type{<:ConstraintKey}) = :constraints
field_for_key(::Type{<:ExpressionKey}) = :expressions
field_for_key(::Type{<:ParameterKey}) = :parameters
field_for_key(::Type{<:InitialConditionKey}) = :initial_conditions

field_for_type(::Type{<:VariableType}) = :variables
field_for_type(::Type{<:AuxVariableType}) = :aux_variables
field_for_type(::Type{<:ConstraintType}) = :constraints
field_for_type(::Type{<:ExpressionType}) = :expressions
field_for_type(::Type{<:ParameterType}) = :parameters
field_for_type(::Type{<:InitialConditionType}) = :initial_conditions

key_for_type(::Type{<:VariableType}) = VariableKey
key_for_type(::Type{<:AuxVariableType}) = AuxVarKey
key_for_type(::Type{<:ConstraintType}) = ConstraintKey
key_for_type(::Type{<:ExpressionType}) = ExpressionKey
key_for_type(::Type{<:ParameterType}) = ParameterKey
key_for_type(::Type{<:InitialConditionType}) = InitialConditionKey

# Store field mapping — differs from field_for_type for ConstraintType,
# which maps to :duals in the store (not :constraints).
store_field_for_type(::Type{<:VariableType}) = :variables
store_field_for_type(::Type{<:AuxVariableType}) = :aux_variables
store_field_for_type(::Type{<:ConstraintType}) = :duals
store_field_for_type(::Type{<:ExpressionType}) = :expressions
store_field_for_type(::Type{<:ParameterType}) = :parameters
