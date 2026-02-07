"""
Minimal mock for AbstractOptimizationContainer.
Implements only the interface required by ModelInternal tests.
"""

# MockContainer subtypes the abstract type from InfrastructureSystems.Optimization
# which is imported as ISOPT in the test module
struct MockContainer <: ISOPT.AbstractOptimizationContainer
    constraints::Dict{Symbol, Any}
end

# Convenience constructor
MockContainer() = MockContainer(Dict{Symbol, Any}())
