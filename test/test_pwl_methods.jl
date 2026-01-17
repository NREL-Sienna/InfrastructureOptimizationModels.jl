"""
Unit tests for piecewise linear (PWL) approximation methods.
Tests the mathematical correctness of breakpoint generation.
"""

# These are loaded by the test harness before this file is included
# using PowerSystems
# using JuMP
using InfrastructureSystems
using Dates

# Define aliases if not already defined by test harness
if !@isdefined(PSI)
    const PSI = PowerOptimizationModels
end
if !@isdefined(PSY)
    using PowerSystems
    const PSY = PowerSystems
end
const IS = InfrastructureSystems
const ISOPT = InfrastructureSystems.Optimization

# Test interpolation variable types
struct TestInterpolationVariable <: PSI.InterpolationVariableType end
struct TestBinaryInterpolationVariable <: PSI.BinaryInterpolationVariableType end

# Define required methods for test types
# Note: Second arg is ::Type{<:PSY.Component} (a type), third arg is a formulation instance
PSI.get_variable_binary(
    ::TestInterpolationVariable,
    ::Type{<:PSY.Component},
    ::PSI.AbstractDeviceFormulation,
) = false
PSI.get_variable_binary(
    ::TestBinaryInterpolationVariable,
    ::Type{<:PSY.Component},
    ::PSI.AbstractDeviceFormulation,
) = true
PSI.get_variable_upper_bound(
    ::TestInterpolationVariable,
    ::PSY.Component,
    ::PSI.AbstractDeviceFormulation,
) = 1.0
PSI.get_variable_lower_bound(
    ::TestInterpolationVariable,
    ::PSY.Component,
    ::PSI.AbstractDeviceFormulation,
) = 0.0
PSI.get_variable_upper_bound(
    ::TestBinaryInterpolationVariable,
    ::PSY.Component,
    ::PSI.AbstractDeviceFormulation,
) = nothing
PSI.get_variable_lower_bound(
    ::TestBinaryInterpolationVariable,
    ::PSY.Component,
    ::PSI.AbstractDeviceFormulation,
) = nothing

# Test formulation type
struct TestPWLFormulation <: PSI.AbstractDeviceFormulation end

#==============================================================================#
# Helper functions
#==============================================================================#

function make_test_thermal(name::String; min_power = 10.0, max_power = 100.0)
    bus = PSY.ACBus(;
        number = 1,
        name = "bus1",
        bustype = PSY.ACBusTypes.PV,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
        available = true,
    )
    return PSY.ThermalStandard(;
        name = name,
        available = true,
        status = true,
        bus = bus,
        active_power = 50.0,
        reactive_power = 0.0,
        rating = max_power,
        active_power_limits = (min = min_power, max = max_power),
        reactive_power_limits = (min = -50.0, max = 50.0),
        ramp_limits = (up = 10.0, down = 10.0),
        time_limits = (up = 2.0, down = 2.0),
        operation_cost = PSY.ThermalGenerationCost(nothing),
        base_power = 100.0,
    )
end

function setup_pwl_test_container(time_steps::UnitRange{Int})
    sys = PSY.System(100.0)
    settings = PSI.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = PSI.OptimizationContainer(sys, settings, JuMP.Model(), PSY.Deterministic)
    PSI.set_time_steps!(container, time_steps)
    return container
end

"""
Test breakpoint generation and return the breakpoints for further assertions.
Checks common invariants: correct count, domain boundaries.
"""
function test_breakpoints(f, min_val, max_val, num_segments)
    x_bkpts, y_bkpts =
        PSI._get_breakpoints_for_pwl_function(min_val, max_val, f; num_segments)
    expected_count = num_segments + 1

    @test length(x_bkpts) == expected_count
    @test length(y_bkpts) == expected_count
    @test x_bkpts[1] ≈ min_val
    @test x_bkpts[end] ≈ max_val

    return x_bkpts, y_bkpts
end

"""
Set up container, add PWL interpolation variables, and return relevant objects.
"""
function setup_and_add_pwl_variables(var_type, devices, time_steps, num_segments)
    container = setup_pwl_test_container(time_steps)
    model = PSI.DeviceModel(PSY.ThermalStandard, TestPWLFormulation)

    PSI.add_sparse_pwl_interpolation_variables!(
        container,
        var_type,
        devices,
        model,
        num_segments,
    )

    var_container = PSI.get_variable(container, var_type, PSY.ThermalStandard)
    jump_model = PSI.get_jump_model(container)

    return (; container, var_container, jump_model)
end

"""
Verify that all expected PWL variables exist and have correct properties.
"""
function verify_pwl_variables(var_container, jump_model, device_names, segment_range,
    time_steps;
    expect_binary, expected_lb = nothing, expected_ub = nothing)
    for name in device_names
        for i in segment_range
            for t in time_steps
                @test haskey(var_container, (name, i, t))
                var = var_container[(name, i, t)]
                @test var isa JuMP.VariableRef
                @test JuMP.is_valid(jump_model, var)
                @test JuMP.is_binary(var) == expect_binary

                if expected_lb !== nothing
                    @test JuMP.lower_bound(var) == expected_lb
                end
                if expected_ub !== nothing
                    @test JuMP.upper_bound(var) == expected_ub
                end
            end
        end
    end
end

#==============================================================================#
# Tests
#==============================================================================#

@testset "PWL Methods" begin
    @testset "_get_breakpoints_for_pwl_function" begin
        @testset "Linear function f(x) = x" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> x, 0.0, 10.0, 5)
            @test y_bkpts ≈ x_bkpts
            @test all(d ≈ 2.0 for d in diff(x_bkpts))
        end

        @testset "Quadratic function f(x) = x^2" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> x^2, 0.0, 4.0, 4)
            @test x_bkpts ≈ [0.0, 1.0, 2.0, 3.0, 4.0]
            @test y_bkpts ≈ [0.0, 1.0, 4.0, 9.0, 16.0]
        end

        @testset "Constant function f(x) = 5" begin
            _, y_bkpts = test_breakpoints(x -> 5.0, 0.0, 10.0, 3)
            @test all(y == 5.0 for y in y_bkpts)
        end

        @testset "Negative domain" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> x^2, -2.0, 2.0, 4)
            @test y_bkpts[1] ≈ y_bkpts[end]  # f(-2) = f(2) = 4
            @test y_bkpts[3] ≈ 0.0           # f(0) = 0
        end

        @testset "Single segment" begin
            x_bkpts, y_bkpts = test_breakpoints(x -> 2x + 1, 0.0, 5.0, 1)
            @test x_bkpts == [0.0, 5.0]
            @test y_bkpts == [1.0, 11.0]
        end

        @testset "Many segments" begin
            x_bkpts, y_bkpts = test_breakpoints(sin, 0.0, Float64(π), 100)
            @test y_bkpts[1] ≈ 0.0 atol = 1e-10
            @test y_bkpts[end] ≈ 0.0 atol = 1e-10
            @test y_bkpts[51] ≈ 1.0 atol = 1e-10  # sin(π/2) = 1
        end

        @testset "Non-integer step sizes" begin
            x_bkpts, _ = test_breakpoints(x -> x, 0.0, 1.0, 3)
            @test x_bkpts ≈ [0.0, 1 / 3, 2 / 3, 1.0]
        end
    end

    @testset "add_sparse_pwl_interpolation_variables!" begin
        @testset "Continuous interpolation variables" begin
            time_steps = 1:3
            num_segments = 4
            devices = [make_test_thermal("gen1"), make_test_thermal("gen2")]

            setup = setup_and_add_pwl_variables(
                TestInterpolationVariable(), devices, time_steps, num_segments,
            )
            @test !isempty(setup.var_container)

            verify_pwl_variables(
                setup.var_container, setup.jump_model,
                PSY.get_name.(devices), 1:num_segments, time_steps;
                expect_binary = false, expected_lb = 0.0, expected_ub = 1.0,
            )
        end

        @testset "Binary interpolation variables" begin
            time_steps = 1:2
            num_segments = 4
            devices = [make_test_thermal("gen1")]

            setup = setup_and_add_pwl_variables(
                TestBinaryInterpolationVariable(), devices, time_steps, num_segments,
            )
            @test !isempty(setup.var_container)

            # Binary variables: num_segments - 1 variables per device
            verify_pwl_variables(
                setup.var_container, setup.jump_model,
                ["gen1"], 1:(num_segments - 1), time_steps;
                expect_binary = true,
            )

            # Should NOT have num_segments-th variable
            @test !haskey(setup.var_container, ("gen1", num_segments, 1))
        end

        @testset "Multiple devices" begin
            time_steps = 1:2
            num_segments = 3
            devices = [make_test_thermal("gen$i") for i in 1:3]

            setup = setup_and_add_pwl_variables(
                TestInterpolationVariable(), devices, time_steps, num_segments,
            )

            for name in PSY.get_name.(devices)
                @test haskey(setup.var_container, (name, 1, 1))
                @test haskey(setup.var_container, (name, num_segments, length(time_steps)))
            end
        end
    end
end

# TODO: tests for _add_generic_incremental_interpolation_constraint!
