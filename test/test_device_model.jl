"""
Unit tests for DeviceModel struct and related functions.
Uses mock types from mocks/mock_components.jl - no PowerSystems dependency.
"""

using Test
using InfrastructureOptimizationModels

# Define aliases if not already defined (mock_components.jl defines PSI)
if !@isdefined(PSI)
    const PSI = InfrastructureOptimizationModels
end

# TestDeviceFormulation, MockThermalGen, MockRenewableGen, MockLoad,
# AbstractMockDevice, AbstractMockGenerator are defined in mocks/mock_components.jl

@testset "DeviceModel" begin
    @testset "Construction with defaults" begin
        model = PSI.DeviceModel(MockThermalGen, TestDeviceFormulation)

        @test PSI.get_component_type(model) == MockThermalGen
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
            MockThermalGen,
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
            MockThermalGen,
            TestDeviceFormulation;
            attributes = Dict{String, Any}("my_key" => "my_value"),
        )

        attrs = PSI.get_attributes(model)
        @test attrs isa Dict{String, Any}
        @test attrs["my_key"] == "my_value"
    end

    @testset "Time series names" begin
        model = PSI.DeviceModel(MockThermalGen, TestDeviceFormulation)
        ts_names = PSI.get_time_series_names(model)
        @test ts_names isa Dict
    end

    @testset "Subsystem management" begin
        model = PSI.DeviceModel(MockThermalGen, TestDeviceFormulation)

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
        @test_throws ArgumentError PSI._check_device_formulation(AbstractMockDevice)
        @test_throws ArgumentError PSI._check_device_formulation(AbstractMockGenerator)

        # Should reject abstract formulation type
        @test_throws ArgumentError PSI._check_device_formulation(
            PSI.AbstractDeviceFormulation,
        )

        # Should accept concrete types
        @test PSI._check_device_formulation(MockThermalGen) === nothing
        @test PSI._check_device_formulation(TestDeviceFormulation) === nothing
    end

    @testset "DeviceModel rejects abstract types in constructor" begin
        # Abstract device type should fail
        @test_throws ArgumentError PSI.DeviceModel(
            AbstractMockDevice,
            TestDeviceFormulation,
        )

        # Abstract formulation type should fail
        @test_throws ArgumentError PSI.DeviceModel(
            MockThermalGen,
            PSI.AbstractDeviceFormulation,
        )
    end

    @testset "set_model!" begin
        dict = Dict{Symbol, Any}()
        model = PSI.DeviceModel(MockThermalGen, TestDeviceFormulation)

        PSI.set_model!(dict, model)

        @test haskey(dict, :MockThermalGen)
        @test dict[:MockThermalGen] === model
    end

    @testset "set_model! warns on overwrite" begin
        dict = Dict{Symbol, Any}()
        model1 = PSI.DeviceModel(MockThermalGen, TestDeviceFormulation)
        model2 = PSI.DeviceModel(MockThermalGen, TestDeviceFormulation)

        PSI.set_model!(dict, model1)

        # Second call should warn about overwriting
        @test_logs (:warn, r"Overwriting.*existing model") PSI.set_model!(dict, model2)
        @test dict[:MockThermalGen] === model2
    end

    @testset "Multiple device types" begin
        # Test with different mock device types
        thermal_model = PSI.DeviceModel(MockThermalGen, TestDeviceFormulation)
        @test PSI.get_component_type(thermal_model) == MockThermalGen

        renewable_model = PSI.DeviceModel(MockRenewableGen, TestDeviceFormulation)
        @test PSI.get_component_type(renewable_model) == MockRenewableGen

        load_model = PSI.DeviceModel(MockLoad, TestDeviceFormulation)
        @test PSI.get_component_type(load_model) == MockLoad
    end

    @testset "FixedOutput formulation" begin
        # FixedOutput is defined in device_model.jl
        model = PSI.DeviceModel(MockThermalGen, PSI.FixedOutput)
        @test PSI.get_formulation(model) == PSI.FixedOutput
    end
end
