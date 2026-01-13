"""
Minimal service mocks.
"""

struct MockReserve
    name::String
    requirement::Float64
    contributing_devices::Vector{Any}
end

get_name(r::MockReserve) = r.name
get_requirement(r::MockReserve) = r.requirement
