#################################################################################
# Standard Variable Types
# These are the base variable types that device formulations use
# PowerSimulations.jl may extend with additional implementations
#################################################################################

# Device Power Variables
struct ActivePowerVariable <: VariableType end
struct ActivePowerInVariable <: VariableType end
struct ActivePowerOutVariable <: VariableType end
struct ReactivePowerVariable <: VariableType end
struct PowerAboveMinimumVariable <: VariableType end

# Device Status Variables
struct OnVariable <: VariableType end
struct StartVariable <: VariableType end
struct StopVariable <: VariableType end
struct HotStartVariable <: VariableType end
struct WarmStartVariable <: VariableType end
struct ColdStartVariable <: VariableType end

# Energy Variables
struct EnergyVariable <: VariableType end

# Reserve Variables
struct ReservationVariable <: VariableType end
struct ActivePowerReserveVariable <: VariableType end
struct ServiceRequirementVariable <: VariableType end

# Auxiliary Variables
struct LiftVariable <: VariableType end

# System Balance Variables
struct SteadyStateFrequencyDeviation <: VariableType end
struct AreaMismatchVariable <: VariableType end
struct DeltaActivePowerUpVariable <: VariableType end
struct DeltaActivePowerDownVariable <: VariableType end
struct AdditionalDeltaActivePowerUpVariable <: VariableType end
struct AdditionalDeltaActivePowerDownVariable <: VariableType end
struct SmoothACE <: VariableType end
struct SystemBalanceSlackUp <: VariableType end
struct SystemBalanceSlackDown <: VariableType end
struct ReserveRequirementSlack <: VariableType end

# Network Variables
struct VoltageMagnitude <: VariableType end
struct VoltageAngle <: VariableType end
struct FlowActivePowerVariable <: VariableType end
struct FlowActivePowerSlackUpperBound <: VariableType end
struct FlowActivePowerSlackLowerBound <: VariableType end
struct FlowActivePowerFromToVariable <: VariableType end
struct FlowActivePowerToFromVariable <: VariableType end
struct FlowReactivePowerFromToVariable <: VariableType end
struct FlowReactivePowerToFromVariable <: VariableType end
struct PhaseShifterAngle <: VariableType end

# Feedforward Slack Variables
struct UpperBoundFeedForwardSlack <: VariableType end
struct LowerBoundFeedForwardSlack <: VariableType end
struct InterfaceFlowSlackUp <: VariableType end
struct InterfaceFlowSlackDown <: VariableType end

# Cost Variables
struct PiecewiseLinearCostVariable <: VariableType end

# Rate Constraint Slack Variables
struct RateofChangeConstraintSlackUp <: VariableType end
struct RateofChangeConstraintSlackDown <: VariableType end

# Contingency Variables
struct PostContingencyActivePowerChangeVariable <: VariableType end
struct PostContingencyActivePowerReserveDeploymentVariable <: VariableType end

# HVDC Variables
struct DCVoltage <: VariableType end
struct DCLineCurrent <: VariableType end
struct ConverterPowerDirection <: VariableType end
struct ConverterCurrent <: VariableType end
struct SquaredConverterCurrent <: VariableType end
struct InterpolationSquaredCurrentVariable <: VariableType end
struct InterpolationBinarySquaredCurrentVariable <: VariableType end
struct ConverterPositiveCurrent <: VariableType end
struct ConverterNegativeCurrent <: VariableType end
struct SquaredDCVoltage <: VariableType end
struct InterpolationSquaredVoltageVariable <: VariableType end
struct InterpolationBinarySquaredVoltageVariable <: VariableType end
struct AuxBilinearConverterVariable <: VariableType end
struct AuxBilinearSquaredConverterVariable <: VariableType end
struct InterpolationSquaredBilinearVariable <: VariableType end
struct InterpolationBinarySquaredBilinearVariable <: VariableType end
struct HVDCFlowDirectionVariable <: VariableType end

#################################################################################
# Standard Expression Types
# These are the base expression types for aggregating terms
#################################################################################

struct SystemBalanceExpressions <: ExpressionType end
struct RangeConstraintLBExpressions <: ExpressionType end
struct RangeConstraintUBExpressions <: ExpressionType end
struct CostExpressions <: ExpressionType end
struct ActivePowerBalance <: ExpressionType end
struct ReactivePowerBalance <: ExpressionType end
struct EmergencyUp <: ExpressionType end
struct EmergencyDown <: ExpressionType end
struct RawACE <: ExpressionType end
struct ProductionCostExpression <: ExpressionType end
struct FuelConsumptionExpression <: ExpressionType end
struct ActivePowerRangeExpressionLB <: ExpressionType end
struct ActivePowerRangeExpressionUB <: ExpressionType end
struct PostContingencyBranchFlow <: ExpressionType end
struct PostContingencyActivePowerGeneration <: ExpressionType end
struct PostContingencyActivePowerBalance <: ExpressionType end
struct NetActivePower <: ExpressionType end
struct DCCurrentBalance <: ExpressionType end
struct HVDCPowerBalance <: ExpressionType end

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
