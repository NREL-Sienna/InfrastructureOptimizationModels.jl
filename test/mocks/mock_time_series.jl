"""
Minimal time series mocks for testing parameter updates.
"""

using Dates

struct MockDeterministic
    name::String
    data::Vector{Float64}
    resolution::Dates.Period
    initial_timestamp::DateTime
end

struct MockSingleTimeSeries
    name::String
    data::Vector{Float64}
    timestamps::Vector{DateTime}
end

get_name(ts::Union{MockDeterministic, MockSingleTimeSeries}) = ts.name
