#################################################################################
# Generic expression infrastructure
# These are generic methods that don't depend on specific device types
#################################################################################

function _ref_index(network_model::NetworkModel{<:AbstractPowerModel}, bus::PSY.ACBus)
    return get_reference_bus(network_model, bus)
end

_get_variable_if_exists(::PSY.MarketBidCost) = nothing
_get_variable_if_exists(cost::PSY.OperationalCost) = PSY.get_variable(cost)

"""
Generic implementation to add expression containers for devices.
"""
function add_expressions!(
    container::OptimizationContainer,
    ::Type{T},
    devices::U,
    model::DeviceModel{D, W},
) where {
    T <: ExpressionType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    W <: AbstractDeviceFormulation,
} where {D <: PSY.Component}
    time_steps = get_time_steps(container)
    names = PSY.get_name.(devices)
    add_expression_container!(container, T(), D, names, time_steps)
    return
end

"""
Specialized implementation for FuelConsumptionExpression that checks for fuel curves.
"""
function add_expressions!(
    container::OptimizationContainer,
    ::Type{T},
    devices::U,
    model::DeviceModel{D, W},
) where {
    T <: FuelConsumptionExpression,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    W <: AbstractDeviceFormulation,
} where {D <: PSY.Component}
    time_steps = get_time_steps(container)
    names = String[]
    found_quad_fuel_functions = false
    for d in devices
        op_cost = PSY.get_operation_cost(d)
        fuel_curve = _get_variable_if_exists(op_cost)
        if fuel_curve isa PSY.FuelCurve
            push!(names, PSY.get_name(d))
            if !found_quad_fuel_functions
                found_quad_fuel_functions =
                    PSY.get_value_curve(fuel_curve) isa PSY.QuadraticCurve
            end
        end
    end

    if !isempty(names)
        expr_type = found_quad_fuel_functions ? JuMP.QuadExpr : GAE
        add_expression_container!(
            container,
            T(),
            D,
            names,
            time_steps;
            expr_type = expr_type,
        )
    end
    return
end

"""
Generic implementation for service models with reserves.
"""
function add_expressions!(
    container::OptimizationContainer,
    ::Type{T},
    devices::U,
    model::ServiceModel{V, W},
) where {
    T <: ExpressionType,
    U <: Union{Vector{D}, IS.FlattenIteratorWrapper{D}},
    V <: PSY.Reserve,
    W <: AbstractReservesFormulation,
} where {D <: PSY.Component}
    time_steps = get_time_steps(container)
    @assert length(devices) == 1
    add_expression_container!(
        container,
        T(),
        D,
        PSY.get_name.(devices),
        time_steps;
        meta = get_service_name(model),
    )
    return
end

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
