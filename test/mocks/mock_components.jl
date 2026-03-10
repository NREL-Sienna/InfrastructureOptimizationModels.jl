"""
Minimal mock components that satisfy PowerSystems device interfaces.
Each mock is ~20 lines and implements only get_name, get_available, etc.

These types can be used:
1. As instance types (creating MockThermalGen instances)
2. As type parameters for DeviceModel{D, B} (replacing PSY.ThermalStandard etc.)
3. As type parameters for container keys (VariableKey, ConstraintKey, etc.)
"""

using InfrastructureOptimizationModels
using InfrastructureSystems
const PSI = InfrastructureOptimizationModels
const IS = InfrastructureSystems

# Mock formulation type for testing DeviceModel
struct TestDeviceFormulation <: PSI.AbstractDeviceFormulation end

abstract type AbstractMockCost end

# Mock operation cost for testing proportional cost functions
struct MockProportionalCost <: AbstractMockCost
    proportional_term::Float64
    is_time_variant::Bool
    fuel_cost::Float64
end

MockProportionalCost(proportional_term::Float64) =
    MockProportionalCost(proportional_term, false, 0.0)
MockProportionalCost(proportional_term::Float64, is_time_variant::Bool) =
    MockProportionalCost(proportional_term, is_time_variant, 0.0)

# FIXME mildly awkward that we need both fixed and fuel_cost here, but otherwise
# can't define get_fuel_cost for MockThermalGen.
struct MockOperationalCost <: AbstractMockCost
    variable::IS.CostCurve
    fixed::Float64
    fuel_cost::Float64
end

get_variable(cost::MockOperationalCost) = cost.variable
get_fixed(cost::MockOperationalCost) = cost.fixed
get_fuel_cost(cost::MockOperationalCost) = cost.fuel_cost

# Abstract mock device type for testing rejection of abstract types in DeviceModel
# Subtypes IS.InfrastructureSystemsComponent so they work with DeviceModel and container keys
abstract type AbstractMockDevice <: IS.InfrastructureSystemsComponent end
abstract type AbstractMockGenerator <: AbstractMockDevice end

# Mock Bus
struct MockBus
    name::String
    number::Int
    bustype::Symbol
end

get_name(b::MockBus) = b.name
get_number(b::MockBus) = b.number
get_bustype(b::MockBus) = b.bustype

# Mock Thermal Generator
struct MockThermalGen <: AbstractMockGenerator
    name::String
    available::Bool
    active_power_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}}
    base_power::Float64
    operation_cost::AbstractMockCost
end

# Constructor with default base_power and no operation cost for backward compatibility
MockThermalGen(name, available, limits) =
    MockThermalGen(name, available, limits, 100.0, MockProportionalCost(0.0))
MockThermalGen(name, available, limits, base_power) =
    MockThermalGen(name, available, limits, base_power, MockProportionalCost(0.0))
MockThermalGen(name, available, limits, base_power, operation_cost::IS.CostCurve) =
    MockThermalGen(
        name,
        available,
        limits,
        base_power,
        MockOperationalCost(operation_cost, 0.0, 0.0),
    )

get_name(g::MockThermalGen) = g.name
get_available(g::MockThermalGen) = g.available
IOM.get_active_power_limits(g::MockThermalGen) = g.active_power_limits
IOM.get_base_power(g::MockThermalGen) = g.base_power
IOM.get_operation_cost(g::MockThermalGen) = g.operation_cost
IS.get_fuel_cost(g::MockThermalGen) = g.operation_cost.fuel_cost

# Mock Renewable Generator
struct MockRenewableGen <: AbstractMockGenerator
    name::String
    available::Bool
    bus::MockBus
    rating::Float64
end

get_name(r::MockRenewableGen) = r.name
get_available(r::MockRenewableGen) = r.available
get_bus(r::MockRenewableGen) = r.bus
get_rating(r::MockRenewableGen) = r.rating

# Mock Load
struct MockLoad <: AbstractMockDevice
    name::String
    available::Bool
    max_active_power::Float64
end

get_name(l::MockLoad) = l.name
get_available(l::MockLoad) = l.available
get_max_active_power(l::MockLoad) = l.max_active_power

# Mock Branch
struct MockBranch <: AbstractMockDevice
    name::String
    available::Bool
    from_bus::MockBus
    to_bus::MockBus
    rating::Float64
    r::Float64
end

MockBranch(name, available, from_bus, to_bus, rating) =
    MockBranch(name, available, from_bus, to_bus, rating, 0.0)

get_name(b::MockBranch) = b.name
get_available(b::MockBranch) = b.available
get_from_bus(b::MockBranch) = b.from_bus
get_to_bus(b::MockBranch) = b.to_bus
get_rate(b::MockBranch) = b.rating
get_r(b::MockBranch) = b.r

# Mock component type for use as type parameter in container keys
# This replaces PSY.ThermalStandard etc. in tests that don't need real PSY types
# Subtypes IS.InfrastructureSystemsComponent so it works with VariableKey, ConstraintKey, etc.
struct MockComponentType <: IS.InfrastructureSystemsComponent end
