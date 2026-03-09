"""
Factory functions for quickly creating test fixtures.
"""

using Dates
using Random

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
    operation_cost = MockProportionalCost(0.0),
)
    return MockThermalGen(name, available, bus, limits, base_power, operation_cost)
end

"""
Generate convex piecewise-linear cost curve points with `n_tranches` segments over [0, pmax].
Returns a vector of (x, y) tuples with strictly increasing slopes.
"""
function _random_convex_pwl_points(n_tranches::Int, pmax::Float64, rng)
    xs = sort(rand(rng, n_tranches - 1)) .* pmax
    points = [(0.0, 0.0)]
    cumulative_cost = 0.0
    prev_x = 0.0
    slope = 5.0 + 20.0 * rand(rng)
    for x in xs
        cumulative_cost += slope * (x - prev_x)
        push!(points, (x, cumulative_cost))
        prev_x = x
        slope += 5.0 + 10.0 * rand(rng)
    end
    cumulative_cost += slope * (pmax - prev_x)
    push!(points, (pmax, cumulative_cost))
    return points
end

"""
Create a mock system on `n` nodes where the graph is connected (and fairly random),
half the nodes have generators, and half have loads.
"""
function make_mock_test_network(n::Int; max_tranches::Int = 4, seed::Int = 42)
    @assert n >= 2 "Need at least 2 nodes for a network"
    rng = MersenneTwister(seed)
    sys = MockSystem(1.0)

    # Create buses
    buses = [MockBus("bus$i", i, :PV) for i in 1:n]
    for bus in buses
        add_component!(sys, bus)
    end

    # Build a connected graph: chain 1-2-3-...-n plus random extra edges
    branch_pairs = Set{Tuple{Int, Int}}()
    perm = shuffle(rng, 1:n)
    for i in 1:(n - 1)
        push!(branch_pairs, (perm[i], perm[i + 1]))
    end
    # Add some random cross-links for variety
    n_extra = div(n, 3)
    for _ in 1:n_extra
        a, b = rand(rng, 1:n), rand(rng, 1:n)
        if a != b
            pair = a < b ? (a, b) : (b, a)
            push!(branch_pairs, pair)
        end
    end
    for (idx, (i, j)) in enumerate(branch_pairs)
        r = 0.005 + 0.005 * rand(rng)
        branch = MockBranch("branch$idx", true, buses[i], buses[j], 1.0, r)
        add_component!(sys, branch)
    end

    # Half the nodes get generators with convex PWL costs, the other half get loads
    n_gens = div(n, 2)
    for i in 1:n_gens
        n_tranches = rand(rng, 2:max_tranches)
        pmax = 1.5 * rand(rng)
        points = _random_convex_pwl_points(n_tranches, pmax, rng)

        pwl = IS.PiecewiseLinearData(points)
        cost_curve = IS.CostCurve(IS.InputOutputCurve(pwl))
        op_cost = MockOperationalCost(cost_curve, 0.0, 0.0)
        gen = MockThermalGen(
            "gen$i", true, buses[i], (min = 0.0, max = pmax), 1.0, op_cost,
        )
        add_component!(sys, gen)
    end
    for i in (n_gens + 1):n
        load = MockLoad("load$(i - n_gens)", true, buses[i], 0.5)
        add_component!(sys, load)
    end

    return sys
end
