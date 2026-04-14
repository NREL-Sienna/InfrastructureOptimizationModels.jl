# McCormick envelope for bilinear products z = x·y.
# Adds 4 linear inequalities that bound z given variable bounds on x and y.

"Standard McCormick envelope constraints bounding the bilinear product z = x·y."
struct McCormickConstraint <: ConstraintType end

"Reformulated McCormick constraints on Bin2 separable variables."
struct ReformulatedMcCormickConstraint <: ConstraintType end

"""
    _mc_setindex!(cons, index, n, constraint)

Helper function for setting constraints by-index in a McCormick constraint container.

Supports 2- and 3-length tuples.
"""
@inline function _mc_setindex!(cons, index::Tuple{A, B}, n::Int, constraint) where {A, B}
    cons[index[1], n, index[2]] = constraint
end

@inline function _mc_setindex!(
    cons,
    index::Tuple{A, B, C},
    n::Int,
    constraint,
) where {A, B, C}
    cons[index[1], index[2], n, index[3]] = constraint
end

function _add_mccormick_envelope!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    z::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64;
    lower_bounds::Bool = true,
)
    if lower_bounds
        _mc_setindex!(
            cons,
            index,
            1,
            JuMP.@constraint(
                jump_model,
                z >= x_min * y + x * y_min - x_min * y_min,
            )
        )
        _mc_setindex!(
            cons,
            index,
            2,
            JuMP.@constraint(
                jump_model,
                z >= x_max * y + x * y_max - x_max * y_max,
            )
        )
    end
    _mc_setindex!(
        cons,
        index,
        3,
        JuMP.@constraint(
            jump_model,
            z <= x_max * y + x * y_min - x_max * y_min,
        )
    )
    _mc_setindex!(
        cons,
        index,
        4,
        JuMP.@constraint(
            jump_model,
            z <= x_min * y + x * y_max - x_min * y_max,
        )
    )
end

function _add_mccormick_envelope!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.VariableRef,
    z::JuMP.VariableRef,
    x_min::Float64,
    x_max::Float64;
    lower_bounds::Bool = true,
)
    _add_mccormick_envelope!(
        jump_model, cons, index,
        x, x, z,
        x_min, x_max, x_min, x_max;
        lower_bounds,
    )
end

# Lower McCormick bounds on (z_p1 − z_x − z_y) for the Bin2 reformulation.
function _add_reformulated_lower_mccormick!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    zp1::JuMP.AbstractJuMPScalar,
    zx::JuMP.AbstractJuMPScalar,
    zy::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    _mc_setindex!(
        cons,
        index,
        1,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy >= 2.0 * (x_min * y + x * y_min - x_min * y_min),
        )
    )
    _mc_setindex!(
        cons,
        index,
        2,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy >= 2.0 * (x_max * y + x * y_max - x_max * y_max),
        )
    )
end

function _add_reformulated_mccormick_bin2!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    zp1::JuMP.AbstractJuMPScalar,
    zx::JuMP.AbstractJuMPScalar,
    zy::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    _add_reformulated_lower_mccormick!(
        jump_model, cons, index, x, y, zp1, zx, zy, x_min, x_max, y_min, y_max,
    )
    # Upper bounds also on (z_p1 − z_x − z_y) since Bin2 has no z_p2
    _mc_setindex!(
        cons,
        index,
        3,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy <= 2.0 * (x_max * y + x * y_min - x_max * y_min),
        )
    )
    _mc_setindex!(
        cons,
        index,
        4,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy <= 2.0 * (x_min * y + x * y_max - x_min * y_max),
        )
    )
end
