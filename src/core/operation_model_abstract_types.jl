"""
Abstract type for Decision Model and Emulation Model. OperationModel structs are parameterized with DecisionProblem or Emulation Problem structs
"""
abstract type OperationModel end

#TODO: Document the required interfaces for custom types
"""
Abstract type for Decision Problems

# Example

import PowerOptimizationModels
const POM = PowerOptimizationModels
struct MyCustomProblem <: POM.DecisionProblem
"""
abstract type DecisionProblem end

"""
Abstract type for Emulation Problems

# Example

import PowerOptimizationModels
const POM = PowerOptimizationModels
struct MyCustomEmulator <: POM.EmulationProblem
"""
abstract type EmulationProblem end

#################################################################################
# Simulation Models Container
# Holds references to models in a simulation
# Used for display/printing purposes
struct SimulationModels
    decision_models::Vector{<:OperationModel}
    emulation_model::Union{Nothing, OperationModel}
end

#################################################################################
# Simulation Sequence
# Holds the execution sequence information for a simulation
# This is a placeholder struct - concrete implementation in PowerSimulations
struct SimulationSequence
    executions_by_model::Dict
    horizons::Dict
    intervals::Dict
    SimulationSequence() = new(Dict(), Dict(), Dict())
end

# Placeholder accessor function for simulation sequence
get_step_resolution(::SimulationSequence) = Dates.Hour(1)

#################################################################################
# Simulation Type
# Abstract type for simulation objects
# Concrete implementation should be in PowerSimulations
abstract type Simulation end

#################################################################################
# Simulation Results Type
# Abstract type for simulation results
# Concrete implementation should be in PowerSimulations
abstract type SimulationResults end

#################################################################################
# Simulation Problem Results Type
# Abstract type for individual problem results within a simulation
# Concrete implementation should be in PowerSimulations
abstract type SimulationProblemResults end
