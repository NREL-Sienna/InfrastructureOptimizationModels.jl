"""
Factory functions for quickly creating test fixtures.
"""

using Dates

"""
Create a mock system with specified number of buses, generators, and loads.
"""
function make_mock_system(;
    n_buses = 3,
    n_gens = 2,
    n_loads = 1,
    base_power = 100.0,
)
    sys = MockSystem(base_power)

    # Create buses
    buses = [MockBus("bus$i", i, :PV) for i in 1:n_buses]
    for bus in buses
        add_component!(sys, bus)
    end

    # Create generators
    for i in 1:n_gens
        gen = MockThermalGen(
            "gen$i",
            true,
            buses[mod1(i, length(buses))],
            (min = 0.0, max = 100.0),
        )
        add_component!(sys, gen)
    end

    # Create loads
    for i in 1:n_loads
        load = MockLoad(
            "load$i",
            true,
            buses[mod1(i, length(buses))],
            50.0,
        )
        add_component!(sys, load)
    end

    return sys
end

"""
Create a mock time series with specified parameters.
"""
function make_mock_time_series(;
    name = "test_ts",
    length = 24,
    resolution = Hour(1),
    initial_timestamp = DateTime(2024, 1, 1),
)
    return MockDeterministic(
        name,
        rand(length),
        resolution,
        initial_timestamp,
    )
end

"""
Create a single mock thermal generator with customizable properties.
"""
function make_mock_thermal(
    name::String;
    available = true,
    bus = MockBus("bus1", 1, :PV),
    limits = (min = 0.0, max = 100.0),
    base_power = 100.0,
    operation_cost = MockOperationCost(0.0),
)
    return MockThermalGen(name, available, bus, limits, base_power, operation_cost)
end
