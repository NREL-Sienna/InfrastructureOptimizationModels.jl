"""
Simple script to verify mock objects work correctly.
Run this to ensure the mock infrastructure is functional.
"""

using Dates

# Load all mocks
include("mocks/mock_system.jl")
include("mocks/mock_components.jl")
include("mocks/mock_time_series.jl")
include("mocks/mock_services.jl")
include("mocks/constructors.jl")

println("✓ All mock files loaded successfully")

# Test MockSystem
sys = MockSystem(100.0)
@assert get_base_power(sys) == 100.0
println("✓ MockSystem works")

# Test MockBus
bus = MockBus("bus1", 1, :PV)
@assert get_name(bus) == "bus1"
@assert get_number(bus) == 1
println("✓ MockBus works")

# Test MockThermalGen
gen = MockThermalGen("gen1", true, bus, (min=10.0, max=100.0))
@assert get_name(gen) == "gen1"
@assert get_available(gen) == true
@assert get_active_power_limits(gen).max == 100.0
println("✓ MockThermalGen works")

# Test factory function
sys2 = make_mock_system(n_buses=3, n_gens=2, n_loads=1)
@assert get_base_power(sys2) == 100.0
buses = get_components(MockBus, sys2)
@assert length(buses) == 3
gens = get_components(MockThermalGen, sys2)
@assert length(gens) == 2
println("✓ make_mock_system() works")

# Test time series
ts = make_mock_time_series(length=24)
@assert length(ts.data) == 24
println("✓ make_mock_time_series() works")

println("\n✅ All mock objects verified successfully!")
println("Mock infrastructure is ready for unit testing.")
