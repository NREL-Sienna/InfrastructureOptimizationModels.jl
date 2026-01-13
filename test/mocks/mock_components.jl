"""
Minimal mock components that satisfy PowerSystems device interfaces.
Each mock is ~20 lines and implements only get_name, get_available, etc.
"""

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
struct MockThermalGen
    name::String
    available::Bool
    bus::MockBus
    active_power_limits::NamedTuple{(:min, :max), Tuple{Float64, Float64}}
end

get_name(g::MockThermalGen) = g.name
get_available(g::MockThermalGen) = g.available
get_bus(g::MockThermalGen) = g.bus
get_active_power_limits(g::MockThermalGen) = g.active_power_limits

# Mock Renewable Generator
struct MockRenewableGen
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
struct MockLoad
    name::String
    available::Bool
    bus::MockBus
    max_active_power::Float64
end

get_name(l::MockLoad) = l.name
get_available(l::MockLoad) = l.available
get_bus(l::MockLoad) = l.bus
get_max_active_power(l::MockLoad) = l.max_active_power

# Mock Branch
struct MockBranch
    name::String
    available::Bool
    from_bus::MockBus
    to_bus::MockBus
    rating::Float64
end

get_name(b::MockBranch) = b.name
get_available(b::MockBranch) = b.available
get_from_bus(b::MockBranch) = b.from_bus
get_to_bus(b::MockBranch) = b.to_bus
get_rate(b::MockBranch) = b.rating
