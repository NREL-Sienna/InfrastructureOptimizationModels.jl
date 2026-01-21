"""
Unit tests for DeviceModel struct and related functions.
Uses TestDeviceFormulation from mocks/mock_components.jl
"""

using PowerSystems
using Test
using InfrastructureOptimizationModels

# Define aliases if not already defined (mock_components.jl defines PSI)
if !@isdefined(PSI)
    const PSI = InfrastructureOptimizationModels
end
if !@isdefined(PSY)
    const PSY = PowerSystems
end

# TestDeviceFormulation is defined in mocks/mock_components.jl

@testset "DeviceModel" begin
    @testset "Construction with defaults" begin
        model = PSI.DeviceModel(PSY.ThermalStandard, TestDeviceFormulation)

        @test PSI.get_component_type(model) == PSY.ThermalStandard
        @test PSI.get_formulation(model) == TestDeviceFormulation
        @test isempty(PSI.get_feedforwards(model))
        @test PSI.get_use_slacks(model) == false
        @test isempty(PSI.get_duals(model))
        @test isempty(PSI.get_services(model))
        @test PSI.get_subsystem(model) === nothing
        @test !PSI.has_service_model(model)
    end

    @testset "Construction with custom values" begin
        model = PSI.DeviceModel(
            PSY.ThermalStandard,
            TestDeviceFormulation;
            use_slacks = true,
            attributes = Dict{String, Any}("custom_attr" => 42),
        )

        @test PSI.get_use_slacks(model) == true
        @test PSI.get_attribute(model, "custom_attr") == 42
        @test PSI.get_attribute(model, "nonexistent") === nothing
    end

    @testset "Attributes merging" begin
        # Custom attributes should merge with defaults from get_default_attributes
        model = PSI.DeviceModel(
            PSY.ThermalStandard,
            TestDeviceFormulation;
            attributes = Dict{String, Any}("my_key" => "my_value"),
        )

        attrs = PSI.get_attributes(model)
        @test attrs isa Dict{String, Any}
        @test attrs["my_key"] == "my_value"
    end

    @testset "Time series names" begin
        model = PSI.DeviceModel(PSY.ThermalStandard, TestDeviceFormulation)
        ts_names = PSI.get_time_series_names(model)
        @test ts_names isa Dict
    end

    @testset "Subsystem management" begin
        model = PSI.DeviceModel(PSY.ThermalStandard, TestDeviceFormulation)

        @test PSI.get_subsystem(model) === nothing

        PSI.set_subsystem!(model, "subsystem_1")
        @test PSI.get_subsystem(model) == "subsystem_1"
    end

    @testset "get_attribute with nothing model" begin
        @test PSI.get_attribute(nothing, "any_key") === nothing
    end

    @testset "get_services with nothing" begin
        @test PSI.get_services(nothing) === nothing
    end

    @testset "_check_device_formulation rejects abstract types" begin
        # Should reject abstract device type
        @test_throws ArgumentError PSI._check_device_formulation(PSY.Device)
        @test_throws ArgumentError PSI._check_device_formulation(PSY.Generator)

        # Should reject abstract formulation type
        @test_throws ArgumentError PSI._check_device_formulation(
            PSI.AbstractDeviceFormulation,
        )

        # Should accept concrete types
        @test PSI._check_device_formulation(PSY.ThermalStandard) === nothing
        @test PSI._check_device_formulation(TestDeviceFormulation) === nothing
    end

    @testset "DeviceModel rejects abstract types in constructor" begin
        # Abstract device type should fail
        @test_throws ArgumentError PSI.DeviceModel(PSY.Device, TestDeviceFormulation)

        # Abstract formulation type should fail
        @test_throws ArgumentError PSI.DeviceModel(
            PSY.ThermalStandard,
            PSI.AbstractDeviceFormulation,
        )
    end

    @testset "_set_model!" begin
        dict = Dict{Symbol, Any}()
        model = PSI.DeviceModel(PSY.ThermalStandard, TestDeviceFormulation)

        PSI._set_model!(dict, model)

        @test haskey(dict, :ThermalStandard)
        @test dict[:ThermalStandard] === model
    end

    @testset "_set_model! warns on overwrite" begin
        dict = Dict{Symbol, Any}()
        model1 = PSI.DeviceModel(PSY.ThermalStandard, TestDeviceFormulation)
        model2 = PSI.DeviceModel(PSY.ThermalStandard, TestDeviceFormulation)

        PSI._set_model!(dict, model1)

        # Second call should warn about overwriting
        @test_logs (:warn, r"Overwriting.*existing model") PSI._set_model!(dict, model2)
        @test dict[:ThermalStandard] === model2
    end

    @testset "Multiple device types" begin
        # Test with different PSY device types
        thermal_model = PSI.DeviceModel(PSY.ThermalStandard, TestDeviceFormulation)
        @test PSI.get_component_type(thermal_model) == PSY.ThermalStandard

        renewable_model = PSI.DeviceModel(PSY.RenewableDispatch, TestDeviceFormulation)
        @test PSI.get_component_type(renewable_model) == PSY.RenewableDispatch

        load_model = PSI.DeviceModel(PSY.PowerLoad, TestDeviceFormulation)
        @test PSI.get_component_type(load_model) == PSY.PowerLoad
    end

    @testset "FixedOutput formulation" begin
        # FixedOutput is defined in device_model.jl
        model = PSI.DeviceModel(PSY.ThermalStandard, PSI.FixedOutput)
        @test PSI.get_formulation(model) == PSI.FixedOutput
    end
end
