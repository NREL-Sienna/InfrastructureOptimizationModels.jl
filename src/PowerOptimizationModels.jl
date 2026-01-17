module PowerOptimizationModels

#################################################################################
# Exports

# Base Models
export DecisionModel
export EmulationModel
export ProblemTemplate
export InitialCondition

# Network Relevant Exports
export NetworkModel
# Note: Concrete network model types (PTDFPowerModel, CopperPlatePowerModel, etc.)
# are defined in PowerSimulations, not here

######## Model Container Types ########
export DeviceModel
export ServiceModel

# Initial Conditions Quantities
export DevicePower
export DeviceStatus
export InitialTimeDurationOn
export InitialTimeDurationOff
export InitialEnergyLevel

# Functions
export build!
## Op Model Exports
export get_initial_conditions
export serialize_problem
export serialize_results
export serialize_optimization_model
## Decision Model Export
export solve!
## Emulation Model Exports
export run!

export set_device_model!
export set_service_model!
export set_network_model!
export get_network_formulation
export get_hvdc_network_model
export set_hvdc_network_model!
## Results interfaces

export get_variable_values
export get_dual_values
export get_parameter_values
export get_aux_variable_values
export get_expression_values
export get_timestamps
export get_model_name
export get_decision_problem_results
export get_emulation_problem_results
export get_system
export get_system!
export set_system!
export list_variable_keys
export list_dual_keys
export list_parameter_keys
export list_aux_variable_keys
export list_expression_keys
export list_variable_names
export list_dual_names
export list_parameter_names
export list_aux_variable_names
export list_expression_names
export list_decision_problems
export list_supported_formats
export load_results!
export read_variable
export read_dual
export read_parameter
export read_aux_variable
export read_expression
export read_variables
export read_duals
export read_parameters
export read_aux_variables
export read_expressions
export read_realized_variable
export read_realized_dual
export read_realized_parameter
export read_realized_aux_variable
export read_realized_expression
export read_realized_variables
export read_realized_duals
export read_realized_parameters
export read_realized_aux_variables
export read_realized_expressions
export get_realized_timestamps
export get_problem_base_power
export get_objective_value
export read_optimizer_stats

## Utils Exports
export OptimizationProblemResults
export OptimizationProblemResultsExport
export OptimizerStats
export get_all_constraint_index
export get_all_variable_index
export get_constraint_index
export get_variable_index
export list_recorder_events
export show_recorder_events
export list_simulation_events
export show_simulation_events
export get_num_partitions

# Key Types (from InfrastructureSystems.Optimization)
export VariableKey
export ConstraintKey
export ParameterKey
export ExpressionKey
export AuxVarKey

# Status Enums (from InfrastructureSystems)
export ModelBuildStatus
export RunStatus

# Variables
export ActivePowerVariable
export ActivePowerInVariable
export ActivePowerOutVariable
export HotStartVariable
export WarmStartVariable
export ColdStartVariable
export EnergyVariable
export LiftVariable
export OnVariable
export ReactivePowerVariable
export ReservationVariable
export ActivePowerReserveVariable
export ServiceRequirementVariable
export StartVariable
export StopVariable
export SteadyStateFrequencyDeviation
export AreaMismatchVariable
export DeltaActivePowerUpVariable
export DeltaActivePowerDownVariable
export AdditionalDeltaActivePowerUpVariable
export AdditionalDeltaActivePowerDownVariable
export SmoothACE
export SystemBalanceSlackUp
export SystemBalanceSlackDown
export ReserveRequirementSlack
export VoltageMagnitude
export VoltageAngle
export FlowActivePowerVariable
export FlowActivePowerSlackUpperBound
export FlowActivePowerSlackLowerBound
export FlowActivePowerFromToVariable
export FlowActivePowerToFromVariable
export FlowReactivePowerFromToVariable
export FlowReactivePowerToFromVariable
export PowerAboveMinimumVariable
export PhaseShifterAngle
export UpperBoundFeedForwardSlack
export LowerBoundFeedForwardSlack
export InterfaceFlowSlackUp
export InterfaceFlowSlackDown
export PiecewiseLinearCostVariable
export RateofChangeConstraintSlackUp
export RateofChangeConstraintSlackDown
export PostContingencyActivePowerChangeVariable
export PostContingencyActivePowerReserveDeploymentVariable
export DCVoltage
export DCLineCurrent
export ConverterPowerDirection
export ConverterCurrent
export SquaredConverterCurrent
export InterpolationSquaredCurrentVariable
export InterpolationBinarySquaredCurrentVariable
export ConverterPositiveCurrent
export ConverterNegativeCurrent
export SquaredDCVoltage
export InterpolationSquaredVoltageVariable
export InterpolationBinarySquaredVoltageVariable
export AuxBilinearConverterVariable
export AuxBilinearSquaredConverterVariable
export InterpolationSquaredBilinearVariable
export InterpolationBinarySquaredBilinearVariable

# Auxiliary variables
export TimeDurationOn
export TimeDurationOff
export PowerOutput
export PowerFlowVoltageAngle
export PowerFlowVoltageMagnitude
export PowerFlowLineReactivePowerFromTo, PowerFlowLineReactivePowerToFrom
export PowerFlowLineActivePowerFromTo, PowerFlowLineActivePowerToFrom
export PowerFlowLossFactors
export PowerFlowVoltageStabilityFactors

# Constraints
export AbsoluteValueConstraint
export ActivePowerVariableTimeSeriesLimitsConstraint
export LineFlowBoundConstraint
export ActivePowerVariableLimitsConstraint
export ActivePowerInVariableTimeSeriesLimitsConstraint
export ActivePowerOutVariableTimeSeriesLimitsConstraint
export ActiveRangeICConstraint
export AreaParticipationAssignmentConstraint
export BalanceAuxConstraint
export CommitmentConstraint
export CopperPlateBalanceConstraint
export DurationConstraint
export EnergyBalanceConstraint
export EqualityConstraint
export FeedforwardSemiContinuousConstraint
export FeedforwardUpperBoundConstraint
export FeedforwardLowerBoundConstraint
export FeedforwardIntegralLimitConstraint
export FlowActivePowerConstraint
export FlowActivePowerFromToConstraint
export FlowActivePowerToFromConstraint
export FlowLimitConstraint
export FlowLimitFromToConstraint
export FlowLimitToFromConstraint
export FlowReactivePowerConstraint
export FlowReactivePowerFromToConstraint
export FlowReactivePowerToFromConstraint
export FrequencyResponseConstraint
export HVDCPowerBalance
export HVDCLosses
export HVDCFlowDirectionVariable
export InputActivePowerVariableLimitsConstraint
export InterfaceFlowLimit
export NetworkFlowConstraint
export NodalBalanceActiveConstraint
export NodalBalanceReactiveConstraint
export OutputActivePowerVariableLimitsConstraint
export PiecewiseLinearCostConstraint
export ParticipationAssignmentConstraint
export ParticipationFractionConstraint
export PhaseAngleControlLimit
export RampConstraint
export RampLimitConstraint
export RangeLimitConstraint
export FlowRateConstraint
export FlowRateConstraintFromTo
export FlowRateConstraintToFrom
export PostContingencyEmergencyRateLimitConstrain
export ReactivePowerVariableLimitsConstraint
export RegulationLimitsConstraint
export RequirementConstraint
export ReserveEnergyCoverageConstraint
export ReservePowerConstraint
export SACEPIDAreaConstraint
export StartTypeConstraint
export StartupInitialConditionConstraint
export StartupTimeLimitTemperatureConstraint
export PostContingencyActivePowerVariableLimitsConstraint
export PostContingencyActivePowerReserveDeploymentVariableLimitsConstraint
export PostContingencyGenerationBalanceConstraint
export PostContingencyRampConstraint
export ImportExportBudgetConstraint
export PiecewiseLinearBlockIncrementalOfferConstraint
export PiecewiseLinearBlockDecrementalOfferConstraint
export NodalBalanceCurrentConstraint
export DCLineCurrentConstraint
export ConverterPowerCalculationConstraint
export ConverterMcCormickEnvelopes
export InterpolationVoltageConstraints
export InterpolationCurrentConstraints
export InterpolationBilinearConstraints
export ConverterLossConstraint
export CurrentAbsoluteValueConstraint

# Parameters
# Time Series Parameters
export ActivePowerTimeSeriesParameter
export ActivePowerOutTimeSeriesParameter
export ActivePowerInTimeSeriesParameter
export ReactivePowerTimeSeriesParameter
export DynamicBranchRatingTimeSeriesParameter
export FuelCostParameter
export PostContingencyDynamicBranchRatingTimeSeriesParameter
export RequirementTimeSeriesParameter
export FromToFlowLimitParameter
export ToFromFlowLimitParameter

# NOTE: Datacenter-specific parameters removed - they belong in DataCenterModels.jl, not here
# If you need these parameters, define them in your domain-specific package that imports PowerOptimizationModels

# Cost Parameters
export CostFunctionParameter

# Feedforward Parameters
export OnStatusParameter
export UpperBoundValueParameter
export LowerBoundValueParameter
export FixValueParameter

# Event Parameters
export AvailableStatusParameter
export AvailableStatusChangeCountdownParameter
export ActivePowerOffsetParameter
export ReactivePowerOffsetParameter

# Expressions
export SystemBalanceExpressions
export RangeConstraintLBExpressions
export RangeConstraintUBExpressions
export CostExpressions
export ActivePowerBalance
export ReactivePowerBalance
export EmergencyUp
export EmergencyDown
export RawACE
export ProductionCostExpression
export FuelConsumptionExpression
export ActivePowerRangeExpressionLB
export ActivePowerRangeExpressionUB
export PostContingencyBranchFlow
export PostContingencyActivePowerGeneration
export PostContingencyActivePowerBalance
export NetActivePower
export DCCurrentBalance

#################################################################################
# Imports
import DataStructures: OrderedDict, Deque, SortedDict
import Logging
import Serialization
# Modeling Imports
import JuMP
# so that users do not need to import JuMP to use a solver with PowerModels
import JuMP: optimizer_with_attributes
import JuMP.Containers: DenseAxisArray, SparseAxisArray
export optimizer_with_attributes
import MathOptInterface
import LinearAlgebra
import JSON3
import PowerSystems
import InfrastructureSystems
import PowerFlows
import PowerNetworkMatrices
import PowerNetworkMatrices: PTDF, VirtualPTDF, LODF, VirtualLODF
export PTDF
export VirtualPTDF
export LODF
export VirtualLODF
import InfrastructureSystems: @assert_op, TableFormat, list_recorder_events, get_name

# IS.Optimization imports: functions that have PSY methods that IS needs to access (therefore necessary)
import InfrastructureSystems.Optimization: get_data_field

# IS.Optimization imports that get reexported: no additional methods in PowerOptimizationModels (therefore necessary)
import InfrastructureSystems.Optimization:
    OptimizationProblemResults, OptimizationProblemResultsExport, OptimizerStats
import InfrastructureSystems.Optimization:
    read_variables, read_duals, read_parameters, read_aux_variables, read_expressions
import InfrastructureSystems.Optimization: get_variable_values, get_dual_values,
    get_parameter_values, get_aux_variable_values, get_expression_values, get_value
import InfrastructureSystems.Optimization:
    get_objective_value, export_realized_results, export_optimizer_stats

# IS.Optimization imports that get reexported: yes additional methods in PowerOptimizationModels (therefore may or may not be desired)
import InfrastructureSystems.Optimization:
    read_variable, read_dual, read_parameter, read_aux_variable, read_expression
import InfrastructureSystems.Optimization: list_variable_keys, list_dual_keys,
    list_parameter_keys, list_aux_variable_keys, list_expression_keys
import InfrastructureSystems.Optimization: list_variable_names, list_dual_names,
    list_parameter_names, list_aux_variable_names, list_expression_names
import InfrastructureSystems.Optimization: read_optimizer_stats, get_optimizer_stats,
    export_results, serialize_results, get_timestamps, get_model_base_power
import InfrastructureSystems.Optimization: get_resolution, get_forecast_horizon

# IS.Optimization imports that stay private, may or may not be additional methods in PowerOptimizationModels
import InfrastructureSystems.Optimization: ArgumentConstructStage, ModelConstructStage
import InfrastructureSystems.Optimization: STORE_CONTAINERS, STORE_CONTAINER_DUALS,
    STORE_CONTAINER_EXPRESSIONS, STORE_CONTAINER_PARAMETERS, STORE_CONTAINER_VARIABLES,
    STORE_CONTAINER_AUX_VARIABLES
import InfrastructureSystems.Optimization: OptimizationContainerKey, VariableKey,
    ConstraintKey, ExpressionKey, AuxVarKey, InitialConditionKey, ParameterKey
import InfrastructureSystems.Optimization:
    RightHandSideParameter, ObjectiveFunctionParameter, TimeSeriesParameter
import InfrastructureSystems.Optimization: VariableType, ConstraintType, AuxVariableType,
    ParameterType, InitialConditionType, ExpressionType
import InfrastructureSystems.Optimization: should_export_variable, should_export_dual,
    should_export_parameter, should_export_aux_variable, should_export_expression
import InfrastructureSystems.Optimization:
    get_entry_type, get_component_type, get_output_dir
import InfrastructureSystems.Optimization: read_results_with_keys, deserialize_key,
    encode_key_as_string, encode_keys_as_strings, should_write_resulting_value,
    convert_result_to_natural_units, to_matrix, get_store_container_type
import InfrastructureSystems.Optimization: get_source_data

# IS.Optimization imports that stay private, may or may not be additional methods in PowerOptimizationModels

# PowerSystems imports
import PowerSystems:
    get_components, get_component, get_available_components, get_available_component,
    get_groups, get_available_groups, stores_time_series_in_memory, get_base_power
import PowerSystems: StartUpStages

export get_name
export get_model_base_power
export get_optimizer_stats
export get_timestamps
export get_resolution

import PowerModels
import TimerOutputs

# Base Imports
import Base.getindex
import Base.isempty
import Base.length
import Base.first
import InteractiveUtils: methodswith

# TimeStamp Management Imports
import Dates
import TimeSeries

# I/O Imports
import DataFrames
import DataFrames: DataFrame, DataFrameRow, Not, innerjoin
import DataFramesMeta: @chain, @orderby, @rename, @select, @subset, @transform
import HDF5
import PrettyTables

# PowerModels exports
export ACPPowerModel
export ACRPowerModel
export ACTPowerModel
export DCPPowerModel
export NFAPowerModel
export DCPLLPowerModel
export LPACCPowerModel
export SOCWRPowerModel
export SOCWRConicPowerModel
export QCRMPowerModel
export QCLSPowerModel

################################################################################

# Type Alias From other Packages
const PM = PowerModels
const PSY = PowerSystems
const POM = PowerOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const MOPFM = MOI.FileFormats.Model
const PNM = PowerNetworkMatrices
const PFS = PowerFlows
const TS = TimeSeries

# Import parameter types from InfrastructureSystems.Optimization
import InfrastructureSystems.Optimization: ParameterType, TimeSeriesParameter

################################################################################

using DocStringExtensions

@template DEFAULT = """
                    $(TYPEDSIGNATURES)
                    $(DOCSTRING)
                    """
# Includes
include("core/definitions.jl")
include("core/time_series_parameter_types.jl")

# Core components
include("core/operation_model_abstract_types.jl")
include("core/network_reductions.jl")
include("core/service_model.jl")
include("core/device_model.jl")
include("core/network_model.jl")
include("core/initial_conditions.jl")
include("core/settings.jl")
include("core/dataset.jl")
include("core/dataset_container.jl")
include("core/results_by_time.jl")

# Order Required
include("core/power_flow_data_wrapper.jl")
include("operation/problem_template.jl")
include("core/optimization_container.jl")
include("core/model_store_params.jl")

# Standard variable and expression types (after OptimizationContainer is defined)
include("core/standard_variables_expressions.jl")

# Common models - extension points for device formulations
include("common_models/variable_properties.jl")
include("common_models/add_variable.jl")
include("common_models/add_auxiliary_variable.jl")
include("common_models/add_parameters.jl")
include("common_models/add_constraints.jl")
include("common_models/add_constraint_dual.jl")
# include("common_models/add_to_expression.jl")
include("common_models/set_expression.jl")
include("common_models/construct_device.jl")
# include("common_models/get_time_series.jl")  # requires TimeSeriesAttributes
include("common_models/objective_function.jl")
include("common_models/add_pwl_methods.jl")
# include("common_models/range_constraint.jl")
# include("common_models/duration_constraints.jl")
# include("common_models/rateofchange_constraints.jl")
include("common_models/get_default_attributes.jl")
include("common_models/add_variable_cost.jl")

# Objective function implementations
# include("objective_function/common.jl")
# include("objective_function/linear_curve.jl")
# include("objective_function/quadratic_curve.jl")
# include("objective_function/piecewise_linear.jl")
# include("objective_function/market_bid.jl")
# include("objective_function/import_export.jl")

include("operation/operation_model_interface.jl")
include("operation/decision_model_store.jl")
include("operation/emulation_model_store.jl")
include("operation/initial_conditions_update_in_memory_store.jl")
include("operation/decision_model.jl")
include("operation/emulation_model.jl")
include("operation/problem_results.jl")
include("operation/operation_model_serialization.jl")
include("operation/time_series_interface.jl")
include("operation/optimization_debugging.jl")
include("operation/model_numerical_analysis_utils.jl")

include("initial_conditions/add_initial_condition.jl")
include("initial_conditions/calculate_initial_condition.jl")

include("initial_conditions/initialization.jl")

# Utils
include("utils/indexing.jl")
@static if pkgversion(PrettyTables).major == 2
    # When PrettyTables v3 is more widely adopted in the ecosystem, we can remove this file.
    # In this case, we should also update the compat bounds in Project.toml to list only
    # PrettyTables v3.
    include("utils/print_pt_v2.jl")
else
    include("utils/print_pt_v3.jl")
end
include("utils/file_utils.jl")
include("utils/logging.jl")
include("utils/dataframes_utils.jl")
include("utils/jump_utils.jl")
include("utils/powersystems_utils.jl")
include("utils/time_series_utils.jl")
include("utils/datetime_utils.jl")
include("utils/generate_valid_formulations.jl")

end
