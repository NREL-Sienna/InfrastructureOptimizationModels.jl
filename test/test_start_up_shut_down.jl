"""
Integration tests for start-up and shut-down cost objective function construction.
Tests the functions in src/objective_function/start_up_shut_down.jl.
Requires PowerSystems types (PSY.ThermalStandard, PSY.ThermalGenerationCost).
Test types defined in test_utils/test_types.jl.
"""

IOM._sos_status(::Type, ::TestDeviceFormulation) = IOM.SOSStatusVariable.NO_VARIABLE

# Formulation already defined in mock_components.jl: TestDeviceFormulation

# Interface implementations for PSY types with our test variable/formulation

IOM.objective_function_multiplier(::TestShutDownVariable, ::TestDeviceFormulation) = 1.0
IOM.objective_function_multiplier(::TestStartVariable, ::TestDeviceFormulation) = 1.0

# Helper to create a PSY ThermalStandard with specified startup/shutdown costs
function make_psy_thermal_with_costs(
    name::String;
    startup_cost::Float64 = 0.0,
    shutdown_cost::Float64 = 0.0,
    must_run::Bool = false,
)
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

    # Create a simple linear cost curve
    cost_curve = PSY.CostCurve(;
        value_curve = IS.LinearCurve(0.0),  # No variable cost for this test
        power_units = PSY.UnitSystem.NATURAL_UNITS,
    )

    op_cost = PSY.ThermalGenerationCost(;
        variable = cost_curve,
        fixed = 0.0,
        start_up = startup_cost,
        shut_down = shutdown_cost,
    )

    return PSY.ThermalStandard(;
        name = name,
        available = true,
        status = true,
        bus = bus,
        active_power = 50.0,
        reactive_power = 10.0,
        rating = 100.0,
        active_power_limits = (min = 10.0, max = 100.0),
        reactive_power_limits = (min = -50.0, max = 50.0),
        ramp_limits = (up = 10.0, down = 10.0),
        time_limits = (up = 2.0, down = 2.0),
        operation_cost = op_cost,
        base_power = 100.0,
        prime_mover_type = PSY.PrimeMovers.ST,
        fuel = PSY.ThermalFuels.COAL,
        must_run = must_run,
    )
end

# Helper to set up container with variables for PSY devices
function setup_startup_shutdown_test_container(
    time_steps::UnitRange{Int},
    devices::Vector{PSY.ThermalStandard},
    var_type::IOM.VariableType;
    resolution = Dates.Hour(1),
)
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = resolution,
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), MockDeterministic)
    IOM.set_time_steps!(container, time_steps)

    # Add variable container for all devices
    device_names = [PSY.get_name(d) for d in devices]
    var_container = IOM.add_variable_container!(
        container,
        var_type,
        PSY.ThermalStandard,
        device_names,
        time_steps,
    )

    # Populate with actual JuMP variables
    jump_model = IOM.get_jump_model(container)
    for name in device_names, t in time_steps
        var_container[name, t] = JuMP.@variable(
            jump_model,
            base_name = "Test_$(name)_$(t)",
        )
    end

    return container
end

# Create a FlattenIteratorWrapper from a vector of PSY devices
function make_psy_device_iterator(devices::Vector{PSY.ThermalStandard})
    return IS.FlattenIteratorWrapper(PSY.ThermalStandard, Vector[devices])
end

@testset "Start-up and Shut-down Cost Objective Functions" begin
    @testset "add_shut_down_cost! adds shutdown cost to objective" begin
        time_steps = 1:3
        shutdown_cost = 50.0
        device = make_psy_thermal_with_costs("gen1"; shutdown_cost = shutdown_cost)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable(),
        )

        devices_iter = make_psy_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container,
            TestShutDownVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        # Verify shutdown costs are in invariant expression (time-invariant case)
        @test verify_objective_coefficients(
            container,
            TestShutDownVariable(),
            PSY.ThermalStandard,
            "gen1",
            shutdown_cost;
            variant = false,
        )
    end

    @testset "add_shut_down_cost! skips must_run devices" begin
        time_steps = 1:2
        shutdown_cost = 50.0
        device = make_psy_thermal_with_costs(
            "gen1";
            shutdown_cost = shutdown_cost,
            must_run = true,
        )
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable(),
        )

        devices_iter = make_psy_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container,
            TestShutDownVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        # must_run device should be skipped - no cost terms added
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_shut_down_cost! with zero cost skips device" begin
        time_steps = 1:2
        device = make_psy_thermal_with_costs("gen1"; shutdown_cost = 0.0)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable(),
        )

        devices_iter = make_psy_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container,
            TestShutDownVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        # Zero cost should be skipped
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_start_up_cost! adds startup cost to objective" begin
        time_steps = 1:3
        startup_cost = 100.0
        device = make_psy_thermal_with_costs("gen1"; startup_cost = startup_cost)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable(),
        )

        devices_iter = make_psy_device_iterator(devices)

        IOM.add_start_up_cost!(
            container,
            TestStartVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        # Verify startup costs are in invariant expression (time-invariant case)
        @test verify_objective_coefficients(
            container,
            TestStartVariable(),
            PSY.ThermalStandard,
            "gen1",
            startup_cost;
            variant = false,
        )
    end

    @testset "add_start_up_cost! skips must_run devices" begin
        time_steps = 1:2
        startup_cost = 100.0
        device = make_psy_thermal_with_costs(
            "gen1";
            startup_cost = startup_cost,
            must_run = true,
        )
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable(),
        )

        devices_iter = make_psy_device_iterator(devices)

        IOM.add_start_up_cost!(
            container,
            TestStartVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        # must_run device should be skipped - no cost terms added
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_start_up_cost! with zero cost skips device" begin
        time_steps = 1:2
        device = make_psy_thermal_with_costs("gen1"; startup_cost = 0.0)
        devices = [device]
        container = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable(),
        )

        devices_iter = make_psy_device_iterator(devices)

        IOM.add_start_up_cost!(
            container,
            TestStartVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        # Zero cost should be skipped
        @test count_objective_terms(container; variant = false) == 0
    end

    @testset "add_start_up_cost! and add_shut_down_cost! with multiple devices" begin
        time_steps = 1:2
        startup1, shutdown1 = 100.0, 50.0
        startup2, shutdown2 = 200.0, 75.0

        device1 = make_psy_thermal_with_costs(
            "gen1";
            startup_cost = startup1,
            shutdown_cost = shutdown1,
        )
        device2 = make_psy_thermal_with_costs(
            "gen2";
            startup_cost = startup2,
            shutdown_cost = shutdown2,
        )
        devices = [device1, device2]

        # Test shutdown costs
        container_sd = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestShutDownVariable(),
        )
        devices_iter = make_psy_device_iterator(devices)

        IOM.add_shut_down_cost!(
            container_sd,
            TestShutDownVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        @test verify_objective_coefficients(
            container_sd,
            TestShutDownVariable(),
            PSY.ThermalStandard,
            "gen1",
            shutdown1;
            variant = false,
        )
        @test verify_objective_coefficients(
            container_sd,
            TestShutDownVariable(),
            PSY.ThermalStandard,
            "gen2",
            shutdown2;
            variant = false,
        )

        # Test startup costs
        container_su = setup_startup_shutdown_test_container(
            time_steps,
            devices,
            TestStartVariable(),
        )
        devices_iter = make_psy_device_iterator(devices)

        IOM.add_start_up_cost!(
            container_su,
            TestStartVariable(),
            devices_iter,
            TestDeviceFormulation(),
        )

        @test verify_objective_coefficients(
            container_su,
            TestStartVariable(),
            PSY.ThermalStandard,
            "gen1",
            startup1;
            variant = false,
        )
        @test verify_objective_coefficients(
            container_su,
            TestStartVariable(),
            PSY.ThermalStandard,
            "gen2",
            startup2;
            variant = false,
        )
    end
end
