# Key types are imported from InfrastructureSystems.Optimization in the main module:
# - AbstractOptimizationContainer, OptimizationKeyType
# - VariableType, ConstraintType, AuxVariableType, ParameterType, InitialConditionType, ExpressionType
# - RightHandSideParameter, ObjectiveFunctionParameter, TimeSeriesParameter
# - ConstructStage, ArgumentConstructStage, ModelConstructStage

# Utility functions for the imported types
convert_result_to_natural_units(::Type{<:VariableType}) = false
convert_result_to_natural_units(::Type{<:ConstraintType}) = false
convert_result_to_natural_units(::Type{<:AuxVariableType}) = false
convert_result_to_natural_units(::Type{<:ExpressionType}) = false
convert_result_to_natural_units(::Type{<:ParameterType}) = false

should_write_resulting_value(::Type{<:VariableType}) = true
should_write_resulting_value(::Type{<:ConstraintType}) = true
should_write_resulting_value(::Type{<:AuxVariableType}) = true
should_write_resulting_value(::Type{<:ExpressionType}) = false
# TODO: Piecewise linear parameter are broken to write
should_write_resulting_value(::Type{<:ParameterType}) = false
