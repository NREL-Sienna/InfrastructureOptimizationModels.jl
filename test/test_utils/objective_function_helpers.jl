"""
Helper functions for testing objective function construction.
Provides utilities for inspecting and verifying objective function coefficients.
"""

using JuMP

"""
Create an OptimizationContainer configured for testing.
Returns container with time_steps already set.
"""
function setup_test_container(
    time_steps::UnitRange{Int};
    base_power = 100.0,
    resolution = Dates.Hour(1),
)
    sys = MockSystem(base_power)
    settings = PSI.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = resolution,
    )
    container = PSI.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    PSI.set_time_steps!(container, time_steps)
    return container
end

"""
Get the coefficient of a variable in the objective function's invariant terms.
Returns 0.0 if the variable is not present.
"""
function get_objective_coefficient(
    container::PSI.OptimizationContainer,
    var_type::PSI.VariableType,
    ::Type{T},
    name::String,
    t::Int,
) where {T}
    obj = PSI.get_objective_expression(container)
    invariant = PSI.get_invariant_terms(obj)
    var = PSI.get_variable(container, var_type, T)[name, t]
    return JuMP.coefficient(invariant, var)
end

"""
Get the coefficient of a variable in the objective function's variant terms.
Returns 0.0 if the variable is not present.
"""
function get_objective_variant_coefficient(
    container::PSI.OptimizationContainer,
    var_type::PSI.VariableType,
    ::Type{T},
    name::String,
    t::Int,
) where {T}
    obj = PSI.get_objective_expression(container)
    variant = PSI.get_variant_terms(obj)
    var = PSI.get_variable(container, var_type, T)[name, t]
    return JuMP.coefficient(variant, var)
end

"""
Verify that objective coefficients match expected values for all time steps.
Checks invariant terms by default.

# Arguments
- `container`: OptimizationContainer to check
- `var_type`: Variable type instance
- `T`: Component type
- `name`: Device name
- `expected`: Either a scalar (same for all time steps) or vector of expected values
- `atol`: Absolute tolerance for comparison (default 1e-10)
- `variant`: If true, check variant terms instead of invariant (default false)

Returns true if all coefficients match within tolerance.
"""
function verify_objective_coefficients(
    container::PSI.OptimizationContainer,
    var_type::PSI.VariableType,
    ::Type{T},
    name::String,
    expected::Union{Float64, Vector{Float64}};
    atol = 1e-10,
    variant = false,
) where {T}
    time_steps = PSI.get_time_steps(container)
    get_coef = variant ? get_objective_variant_coefficient : get_objective_coefficient

    for t in time_steps
        exp_val = expected isa Vector ? expected[t] : expected
        actual = get_coef(container, var_type, T, name, t)
        if !isapprox(actual, exp_val; atol = atol)
            @warn "Coefficient mismatch at t=$t: expected $exp_val, got $actual"
            return false
        end
    end
    return true
end

"""
Get the total number of terms in the objective function's invariant expression.
Useful for verifying that the expected number of cost terms were added.
"""
function count_objective_terms(container::PSI.OptimizationContainer; variant = false)
    obj = PSI.get_objective_expression(container)
    expr = variant ? PSI.get_variant_terms(obj) : PSI.get_invariant_terms(obj)
    if expr isa JuMP.GenericAffExpr
        return length(expr.terms)
    elseif expr isa JuMP.GenericQuadExpr
        return length(expr.aff.terms) + length(expr.terms)
    else
        return 0
    end
end

"""
Get the quadratic coefficient of a variable (coefficient of var^2) in the objective.
Returns 0.0 if the variable is not present in quadratic terms.
"""
function get_objective_quadratic_coefficient(
    container::PSI.OptimizationContainer,
    var_type::PSI.VariableType,
    ::Type{T},
    name::String,
    t::Int,
) where {T}
    obj = PSI.get_objective_expression(container)
    invariant = PSI.get_invariant_terms(obj)
    var = PSI.get_variable(container, var_type, T)[name, t]
    # JuMP.coefficient(expr, var, var) gets the coefficient of var^2
    return JuMP.coefficient(invariant, var, var)
end

"""
Verify that quadratic objective coefficients match expected values for all time steps.
Checks both linear and quadratic coefficients.

# Arguments
- `container`: OptimizationContainer to check
- `var_type`: Variable type instance
- `T`: Component type
- `name`: Device name
- `expected_linear`: Either a scalar or vector of expected linear coefficients
- `expected_quadratic`: Either a scalar or vector of expected quadratic coefficients
- `atol`: Absolute tolerance for comparison (default 1e-10)

Returns true if all coefficients match within tolerance.
"""
function verify_quadratic_objective_coefficients(
    container::PSI.OptimizationContainer,
    var_type::PSI.VariableType,
    ::Type{T},
    name::String,
    expected_linear::Union{Float64, Vector{Float64}},
    expected_quadratic::Union{Float64, Vector{Float64}};
    atol = 1e-10,
) where {T}
    time_steps = PSI.get_time_steps(container)

    for t in time_steps
        exp_lin = expected_linear isa Vector ? expected_linear[t] : expected_linear
        exp_quad =
            expected_quadratic isa Vector ? expected_quadratic[t] : expected_quadratic

        actual_lin = get_objective_coefficient(container, var_type, T, name, t)
        actual_quad = get_objective_quadratic_coefficient(container, var_type, T, name, t)

        if !isapprox(actual_lin, exp_lin; atol = atol)
            @warn "Linear coefficient mismatch at t=$t: expected $exp_lin, got $actual_lin"
            return false
        end
        if !isapprox(actual_quad, exp_quad; atol = atol)
            @warn "Quadratic coefficient mismatch at t=$t: expected $exp_quad, got $actual_quad"
            return false
        end
    end
    return true
end
