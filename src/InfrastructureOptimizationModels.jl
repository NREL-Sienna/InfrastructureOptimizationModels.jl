module InfrastructureOptimizationModels

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
import MathOptInterface
import LinearAlgebra
import JSON3
import PowerSystems
import InfrastructureSystems
import PowerNetworkMatrices
import PowerNetworkMatrices: PTDF, VirtualPTDF, LODF, VirtualLODF
import InfrastructureSystems: @assert_op, TableFormat, list_recorder_events, get_name
import InfrastructureSystems:
    get_value_curve, get_power_units, get_function_data, get_proportional_term,
    get_quadratic_term, get_fuel_cost

# IS.Optimization imports: base types that remain in InfrastructureSystems
# Note: ModelBuildStatus is aliased in definitions.jl, so don't import it directly
# TODO: some of these are device specific enough to belong in POM.
import InfrastructureSystems.Optimization:
    AbstractOptimizationContainer,
    OptimizationKeyType,
    AbstractModelStoreParams,
    # Key types - imported from IS.Optimization to avoid duplication
    VariableType,
    ConstraintType,
    AuxVariableType,
    ParameterType,
    InitialConditionType,
    ExpressionType,
    RightHandSideParameter,
    ObjectiveFunctionParameter,
    TimeSeriesParameter,
    ConstructStage,
    ArgumentConstructStage,
    ModelConstructStage,
    # Formulation abstract types
    AbstractDeviceFormulation,
    AbstractServiceFormulation,
    AbstractReservesFormulation,
    AbstractThermalFormulation,
    AbstractRenewableFormulation,
    AbstractStorageFormulation,
    AbstractLoadFormulation,
    AbstractHVDCNetworkModel,
    AbstractPowerModel,
    AbstractPTDFModel,
    AbstractSecurityConstrainedPTDFModel,
    AbstractActivePowerModel,
    AbstractACPowerModel,
    AbstractACPModel,
    ACPPowerModel,
    AbstractPowerFlowEvaluationModel,
    AbstractPowerFlowEvaluationData

import InfrastructureSystems:
    @scoped_enum,
    TableFormat,
    get_variables,
    get_parameters,
    get_total_cost,
    get_optimizer_stats,
    get_timestamp,
    write_results,
    get_source_data,
    configure_logging,
    strip_module_name,
    to_namedtuple,
    get_uuid,
    compute_file_hash,
    convert_for_path,
    COMPONENT_NAME_DELIMITER,
    # Additional imports needed by core optimization files
    InfrastructureSystemsType,
    InfrastructureSystemsComponent,
    Results,
    TimeSeriesCacheKey,
    TimeSeriesCache,
    InvalidValue,
    ConflictingInputsError

# PowerSystems imports
import PowerSystems:
    get_components,
    get_component,
    get_available_components,
    get_available_component,
    get_groups,
    get_available_groups,
    stores_time_series_in_memory,
    get_base_power,
    get_active_power_limits,
    get_start_up,
    get_shut_down,
    get_must_run,
    get_operation_cost
import PowerSystems: StartUpStages

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
import CSV
import DataFrames
import DataFrames: DataFrame, DataFrameRow, Not, innerjoin, select
import DataFramesMeta: @chain, @orderby, @rename, @select, @subset, @transform
import HDF5
import PrettyTables

################################################################################
# Type Aliases

const PSY = PowerSystems
const POM = InfrastructureOptimizationModels
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization
const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const MOPFM = MOI.FileFormats.Model
const PNM = PowerNetworkMatrices
const TS = TimeSeries

################################################################################

using DocStringExtensions

@template DEFAULT = """
                    $(TYPEDSIGNATURES)
                    $(DOCSTRING)
                    """

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
export FixedOutput

# Parameter Container Infrastructure
export ParameterContainer
export ParameterAttributes
export NoAttributes
export TimeSeriesAttributes
export VariableValueAttributes
export CostFunctionAttributes
export EventParametersAttributes
export ValidDataParamEltypes

# Initial Conditions Quantities
export DevicePower
export DeviceStatus
export InitialTimeDurationOn
export InitialTimeDurationOff
export InitialEnergyLevel

# Functions
export build!
export validate_time_series!
export init_optimization_container!
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

# Extension points for downstream packages (e.g., PowerOperationsModels)
# These functions have fallback implementations in IOM but are meant to be
# extended with device-specific methods in POM
export construct_device!
export construct_service!
export add_variables!
export add_constraints!
export add_to_expression!
export add_constant_to_jump_expression!
export add_proportional_to_jump_expression!
export add_linear_to_jump_expression!
# Cost term helpers (generic objective function building blocks)
export add_cost_term_invariant!
export add_cost_term_variant!
export add_pwl_variables!
export add_pwl_linking_constraint!
export add_pwl_normalization_constraint!
export add_pwl_sos2_constraint!
export get_pwl_cost_expression
export objective_function!
export initial_condition_variable
export initial_condition_default
export process_market_bid_parameters!

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
export get_num_partitions
export jump_value
export ConstraintBounds
export VariableBounds

# Internal accessors needed by downstream packages
export get_network_model
export get_value
export get_initial_conditions_data
export get_initial_condition_value
export get_objective_expression
export get_formulation
export get_settings
export get_rebuild_model

# Expression infrastructure (needed by add_to_expression.jl implementations)
export get_parameter
export get_parameter_array
export get_network_reduction
export get_multiplier_array
export get_parameter_column_refs
export get_service_name
export get_default_time_series_type
export get_expression_multiplier
export get_variable_multiplier
export get_multiplier_value
export add_expression_container!

# Initial condition infrastructure (extension points for POM)
export get_initial_conditions_value
export update_initial_conditions!
# Note: TimeDurationOn and TimeDurationOff are device-specific and defined in POM

# Key Types (defined in IOM)
export OptimizationContainerKey
export VariableKey
export ConstraintKey
export ParameterKey
export ExpressionKey
export AuxVarKey

# Abstract Key Types (from InfrastructureSystems.Optimization)
export VariableType
export ConstraintType
export AuxVariableType
export ParameterType
export InitialConditionType
export ExpressionType

# Standard Expression Types (abstract and concrete)
export SystemBalanceExpressions
export RangeConstraintLBExpressions
export RangeConstraintUBExpressions
export CostExpressions
export PostContingencyExpressions
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
export NetActivePower
export DCCurrentBalance
export HVDCPowerBalance

# Abstract types for extensions (from InfrastructureSystems.Optimization)
export AbstractPowerFlowEvaluationData

# Status Enums (from InfrastructureSystems)
export ModelBuildStatus
export RunStatus
export SimulationBuildStatus

# Problem Types
export DefaultDecisionProblem
export DefaultEmulationProblem

# Settings and Data Types
export Settings
export InitialConditionsData

# Constants
export COST_EPSILON
export INITIALIZATION_PROBLEM_HORIZON_COUNT

# Re-exports from imports
export optimizer_with_attributes
export PTDF
export VirtualPTDF
export LODF
export VirtualLODF
export get_name
export get_model_base_power
export get_optimizer_stats
export get_timestamps
export get_resolution

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

#################################################################################
# Includes
# NOTE: all tracked files are either included here, or have a commented-out include.

# Core optimization types must come first
include("core/optimization_container_types.jl")       # Abstract types (VariableType, etc.)
include("core/definitions.jl")                        # Aliases and enums (needs VariableType)
include("core/optimization_container_keys.jl")        # Keys depend on types
include("core/parameter_container.jl")                # Parameter container infrastructure
include("core/abstract_model_store.jl")               # Store depends on keys
include("core/optimizer_stats.jl")                    # Stats standalone
include("core/optimization_container_metadata.jl")    # Metadata depends on keys
include("core/optimization_problem_results_export.jl") # Export config
include("core/optimization_problem_results.jl")       # Results depends on all above
include("core/model_internal.jl")                     # Internal state (needs ModelBuildStatus)

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
include("operation/problem_template.jl")
include("core/optimization_container.jl")
include("core/model_store_params.jl")

# Standard variable and expression types (after OptimizationContainer is defined)
include("core/standard_variables_expressions.jl")

# Common models - extension points for device formulations
include("common_models/interfaces.jl")
include("common_models/add_variable.jl")
include("common_models/add_auxiliary_variable.jl")
include("common_models/add_constraint_dual.jl")
include("common_models/add_jump_expressions.jl") # helpers only used in POM.
include("common_models/set_expression.jl") # helpers only used in POM.
include("common_models/get_time_series.jl")
include("common_models/add_pwl_methods.jl")
include("common_models/constraint_helpers.jl")
# include("common_models/range_constraint.jl")
# include("common_models/duration_constraints.jl")
# include("common_models/rateofchange_constraints.jl")

# Objective function implementations
include("objective_function/cost_term_helpers.jl") # generic helpers: add_cost_term_{invariant,variant}!, PWL helpers
include("objective_function/common.jl")
include("objective_function/proportional.jl") # add_proportional_cost! and add_proportional_cost_maybe_time_variant!
include("objective_function/start_up_shut_down.jl") # add_{start_up, shut_down}_cost!
# add_variable_cost_to_objective! implementations and that's it (no other exported functions)
# same 5 arguments: container, variable, component, cost_curve, formulation.
include("objective_function/linear_curve.jl")
include("objective_function/quadratic_curve.jl")
include("objective_function/import_export.jl")

# add_variable_cost! implementations, but "it's complicated." Other stuff exported too
include("objective_function/piecewise_linear.jl")
# this one is a mess.
# include("objective_function/market_bid.jl")

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
