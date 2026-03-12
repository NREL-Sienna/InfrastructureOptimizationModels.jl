# McCormick envelope for bilinear products z = x·y.
# Adds 4 linear inequalities that bound z given variable bounds on x and y.

struct McCormickConstraint <: ConstraintType end

"""
    _add_mccormick_envelope!(container, C, names, time_steps, x_var, y_var, z_var, x_min, x_max, y_min, y_max, meta)

Add McCormick envelope constraints for the bilinear product z ≈ x·y.

For each (name, t), adds 4 linear inequalities:
```
z ≥ x_min·y + x·y_min − x_min·y_min
z ≥ x_max·y + x·y_max − x_max·y_max
z ≤ x_max·y + x·y_min − x_max·y_min
z ≤ x_min·y + x·y_max − x_min·y_max
```

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `z_var`: container of z variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `meta::String`: identifier for container keys

# Returns
- Nothing. Constraints are added in-place.
"""
function _add_mccormick_envelope!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    z_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op y_max > y_min
    jump_model = get_jump_model(container)

    mc_cons = @_add_container(constraints, McCormickConstraint, 1:4, sparse)

    for name in names, t in time_steps
        x = x_var[name, t]
        y = y_var[name, t]
        z = z_var[name, t]

        # z ≥ x_min·y + x·y_min − x_min·y_min
        mc_cons[(name, 1, t)] = JuMP.@constraint(
            jump_model,
            z >= x_min * y + x * y_min - x_min * y_min,
        )
        # z ≥ x_max·y + x·y_max − x_max·y_max
        mc_cons[(name, 2, t)] = JuMP.@constraint(
            jump_model,
            z >= x_max * y + x * y_max - x_max * y_max,
        )
        # z ≤ x_max·y + x·y_min − x_max·y_min
        mc_cons[(name, 3, t)] = JuMP.@constraint(
            jump_model,
            z <= x_max * y + x * y_min - x_max * y_min,
        )
        # z ≤ x_min·y + x·y_max − x_min·y_max
        mc_cons[(name, 4, t)] = JuMP.@constraint(
            jump_model,
            z <= x_min * y + x * y_max - x_min * y_max,
        )
    end

    return
end
