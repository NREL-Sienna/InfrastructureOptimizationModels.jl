#################################################################################
# Standard Variable Types
# Only types that IOM's own infrastructure code references belong here.
# Device-specific variable types are defined in PowerOperationsModels.jl.
#################################################################################

# Device Power Variables (used in objective_function/, rateofchange_constraints)
struct ActivePowerVariable <: VariableType end
struct ActivePowerInVariable <: VariableType end
struct ActivePowerOutVariable <: VariableType end
struct PowerAboveMinimumVariable <: VariableType end

# Device Status Variables (used in range_constraint, objective_function/)
struct OnVariable <: VariableType end
struct StartVariable <: VariableType end
struct StopVariable <: VariableType end

# Service Variables (used in market_bid)
struct ServiceRequirementVariable <: VariableType end

# Cost Variables (used in piecewise_linear)
struct PiecewiseLinearCostVariable <: SparseVariableType end

# Rate Constraint Slack Variables (used in rateofchange_constraints)
struct RateofChangeConstraintSlackUp <: VariableType end
struct RateofChangeConstraintSlackDown <: VariableType end

# HVDC Variables (used in add_pwl_methods)
struct DCVoltage <: VariableType end

#################################################################################
# Standard Expression Types
# These are the base expression types for aggregating terms
#################################################################################

# Abstract types for expression hierarchies (used in IOM infrastructure code)
abstract type SystemBalanceExpressions <: ExpressionType end
abstract type RangeConstraintLBExpressions <: ExpressionType end
abstract type RangeConstraintUBExpressions <: ExpressionType end
abstract type CostExpressions <: ExpressionType end
abstract type PostContingencyExpressions <: ExpressionType end

# Concrete expression types used in IOM code
struct ProductionCostExpression <: CostExpressions end
struct FuelConsumptionExpression <: ExpressionType end
struct ActivePowerRangeExpressionLB <: RangeConstraintLBExpressions end
struct ActivePowerRangeExpressionUB <: RangeConstraintUBExpressions end

# Concrete expression types defined here for POM (not used in IOM code directly,
# but IOM exports them and POM relies on getting them from IOM)
struct ActivePowerBalance <: SystemBalanceExpressions end
struct ReactivePowerBalance <: SystemBalanceExpressions end
struct EmergencyUp <: ExpressionType end
struct EmergencyDown <: ExpressionType end
struct RawACE <: ExpressionType end
struct PostContingencyBranchFlow <: PostContingencyExpressions end
struct PostContingencyActivePowerGeneration <: PostContingencyExpressions end
struct NetActivePower <: ExpressionType end
struct DCCurrentBalance <: ExpressionType end
struct HVDCPowerBalance <: ExpressionType end

# Result writing configuration for expression types
should_write_resulting_value(::Type{<:CostExpressions}) = true
should_write_resulting_value(::Type{FuelConsumptionExpression}) = true
should_write_resulting_value(::Type{RawACE}) = true
should_write_resulting_value(::Type{ActivePowerBalance}) = true
should_write_resulting_value(::Type{ReactivePowerBalance}) = true
should_write_resulting_value(::Type{DCCurrentBalance}) = true

# ProductionCostExpression-specific container method (moved here from optimization_container.jl
# because it requires ProductionCostExpression to be defined first)
function add_expression_container!(
    container::OptimizationContainer,
    ::T,
    ::Type{U},
    axs...;
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {T <: ProductionCostExpression, U <: Union{PSY.Component, PSY.System}}
    expr_key = ExpressionKey(T, U, meta)
    expr_type = JuMP.QuadExpr
    return _add_expression_container!(
        container,
        expr_key,
        expr_type,
        axs...;
        sparse = sparse,
    )
end

#################################################################################
# Base Methods
#################################################################################

"""
    requires_initialization(formulation::AbstractDeviceFormulation)

Check if a device formulation requires initial conditions.
Default implementation returns false. Override for formulations with state variables.
"""
function requires_initialization(::AbstractDeviceFormulation)
    return false
end

"""
    add_to_expression!(
        container::OptimizationContainer,
        expression_type::Type{<:ExpressionType},
        variable_type::Type{<:VariableType},
        devices,
        model::DeviceModel,
        network_model::NetworkModel,
    )

Add device variables to system-wide expression.
This is a generic fallback that errors - specific implementations should override.
"""
function add_to_expression!(
    container::OptimizationContainer,
    expression_type::Type{<:ExpressionType},
    variable_type::Type{<:VariableType},
    devices,
    model::DeviceModel,
    network_model::NetworkModel,
)
    error(
        "add_to_expression! not implemented for expression_type=$expression_type, variable_type=$variable_type, device_type=$(typeof(devices.values[1]))",
    )
end
