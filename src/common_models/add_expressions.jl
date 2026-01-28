#################################################################################
# JuMP expression helpers
# These wrap JuMP.add_to_expression! with consistent patterns
# Named to clarify their different purposes:
# - add_constant_to_jump_expression!: adds a single constant value
# - add_proportional_to_jump_expression!: adds multiplier * variable (or parameter * multiplier)
# - add_linear_to_jump_expression!: adds constant + multiplier * variable
#################################################################################

"""
Add constant value to JuMP expression.
"""
function add_constant_to_jump_expression!(
    expression::T,
    value::Float64,
) where {T <: JuMP.AbstractJuMPScalar}
    JuMP.add_to_expression!(expression, value)
    return
end

"""
Add variable with multiplier to JuMP expression: expression += multiplier * var
"""
function add_proportional_to_jump_expression!(
    expression::T,
    var::JuMP.VariableRef,
    multiplier::Float64,
) where {T <: JuMP.AbstractJuMPScalar}
    JuMP.add_to_expression!(expression, multiplier, var)
    return
end

"""
Add product of parameter and multiplier to JuMP expression: expression += parameter * multiplier
"""
function add_proportional_to_jump_expression!(
    expression::T,
    parameter::Float64,
    multiplier::Float64,
) where {T <: JuMP.AbstractJuMPScalar}
    add_constant_to_jump_expression!(expression, parameter * multiplier)
    return
end

"""
Add affine term to JuMP expression: expression += constant + multiplier * var
"""
function add_linear_to_jump_expression!(
    expression::T,
    var::JuMP.VariableRef,
    multiplier::Float64,
    constant::Float64,
) where {T <: JuMP.AbstractJuMPScalar}
    add_constant_to_jump_expression!(expression, constant)
    add_proportional_to_jump_expression!(expression, var, multiplier)
    return
end
