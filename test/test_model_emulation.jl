# FIXME not working and not included in the tests. integration of emulation models in
# POM-IOM split is a work in progress.
@testset "Emulation Model Build" begin
    template = get_thermal_dispatch_template_network()
    c_sys5 = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc";
        add_single_time_series = true,
        force_build = true,
    )

    model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    template = get_thermal_standard_uc_template()
    c_sys5_uc_re = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc_re";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, RenewableDispatch, RenewableFullDispatch)
    model = EmulationModel(template, c_sys5_uc_re; optimizer = HiGHS_optimizer)

    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
    @test !isempty(collect(readdir(PSI.get_recorder_dir(model))))
end

@testset "Emulation Model initial_conditions test for ThermalGen" begin
    ######## Test with ThermalStandardUnitCommitment ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc_re = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc_re";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, RenewableDispatch, RenewableFullDispatch)
    model = EmulationModel(template, c_sys5_uc_re; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT
    check_duration_on_initial_conditions_values(model, ThermalStandard)
    check_duration_off_initial_conditions_values(model, ThermalStandard)
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with ThermalMultiStartUnitCommitment ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, ThermalMultiStart, ThermalMultiStartUnitCommitment)
    model = EmulationModel(template, c_sys5_uc; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 1, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT

    check_duration_on_initial_conditions_values(model, ThermalStandard)
    check_duration_off_initial_conditions_values(model, ThermalStandard)
    check_duration_on_initial_conditions_values(model, ThermalMultiStart)
    check_duration_off_initial_conditions_values(model, ThermalMultiStart)
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with ThermalStandardUnitCommitment ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, ThermalMultiStart, ThermalStandardUnitCommitment)
    model = EmulationModel(template, c_sys5_uc; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 1, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT
    check_duration_on_initial_conditions_values(model, ThermalStandard)
    check_duration_off_initial_conditions_values(model, ThermalStandard)
    check_duration_on_initial_conditions_values(model, ThermalMultiStart)
    check_duration_off_initial_conditions_values(model, ThermalMultiStart)
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with ThermalStandardDispatch ########
    template = get_thermal_standard_uc_template()
    c_sys5_uc = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    device_model = DeviceModel(PSY.ThermalStandard, PSI.ThermalStandardDispatch)
    set_device_model!(template, device_model)
    model = EmulationModel(template, c_sys5_uc; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT
end

@testset "Emulation Model initial_conditions test for Hydro" begin
    ######## Test with HydroDispatchRunOfRiver ########
    template = get_thermal_dispatch_template_network()
    c_sys5_hyd = PSB.build_system(
        PSITestSystems,
        "c_sys5_hyd";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, HydroDispatch, HydroDispatchRunOfRiver)
    set_device_model!(template, HydroTurbine, HydroTurbineEnergyDispatch)
    set_device_model!(template, HydroReservoir, HydroEnergyModelReservoir)
    model = EmulationModel(template, c_sys5_hyd; optimizer = HiGHS_optimizer)
    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT
    initial_conditions_data =
        PSI.get_initial_conditions_data(PSI.get_optimization_container(model))
    @test !PSI.has_initial_condition_value(
        initial_conditions_data,
        ActivePowerVariable(),
        HydroTurbine,
    )
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    ######## Test with HydroCommitmentRunOfRiver ########
    template = get_thermal_dispatch_template_network()
    c_sys5_hyd = PSB.build_system(
        PSITestSystems,
        "c_sys5_hyd";
        add_single_time_series = true,
        force_build = true,
    )
    set_device_model!(template, HydroDispatch, HydroCommitmentRunOfRiver)
    set_device_model!(template, HydroTurbine, HydroTurbineEnergyCommitment)
    set_device_model!(template, HydroReservoir, HydroEnergyModelReservoir)
    model = EmulationModel(template, c_sys5_hyd; optimizer = HiGHS_optimizer)

    @test build!(model; executions = 10, output_dir = mktempdir(; cleanup = true)) ==
          PSI.ModelBuildStatus.BUILT
    initial_conditions_data =
        PSI.get_initial_conditions_data(PSI.get_optimization_container(model))
    @test PSI.has_initial_condition_value(
        initial_conditions_data,
        OnVariable(),
        HydroTurbine,
    )
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
end

@testset "Emulation Model Outputs" begin
    template = get_thermal_dispatch_template_network()
    c_sys5 = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc";
        add_single_time_series = true,
        force_build = true,
    )

    model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
    executions = 10
    @test build!(
        model;
        executions = executions,
        output_dir = mktempdir(; cleanup = true),
    ) ==
          PSI.ModelBuildStatus.BUILT
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
    outputs = OptimizationProblemOutputs(model)
    @test list_aux_variable_names(outputs) == []
    @test list_aux_variable_keys(outputs) == []
    @test list_variable_names(outputs) == ["ActivePowerVariable__ThermalStandard"]
    @test list_variable_keys(outputs) ==
          [PSI.VariableKey(ActivePowerVariable, ThermalStandard)]
    @test list_dual_names(outputs) == []
    @test list_dual_keys(outputs) == []
    @test list_parameter_names(outputs) == ["ActivePowerTimeSeriesParameter__PowerLoad"]
    @test list_parameter_keys(outputs) ==
          [PSI.ParameterKey(ActivePowerTimeSeriesParameter, PowerLoad)]

    @test read_variable(outputs, "ActivePowerVariable__ThermalStandard") isa DataFrame
    @test read_variable(outputs, ActivePowerVariable, ThermalStandard) isa DataFrame
    @test read_variable(
        outputs,
        PSI.VariableKey(ActivePowerVariable, ThermalStandard),
    ) isa
          DataFrame

    @test read_parameter(outputs, "ActivePowerTimeSeriesParameter__PowerLoad") isa DataFrame
    @test read_parameter(outputs, ActivePowerTimeSeriesParameter, PowerLoad) isa DataFrame
    @test read_parameter(
        outputs,
        PSI.ParameterKey(ActivePowerTimeSeriesParameter, PowerLoad),
    ) isa DataFrame

    @test read_optimizer_stats(model) isa DataFrame
    for n in names(read_optimizer_stats(model))
        stats_values = read_optimizer_stats(model)[!, n]
        if any(ismissing.(stats_values))
            @test ismissing.(stats_values) ==
                  ismissing.(read_optimizer_stats(outputs)[!, n])
        elseif any(isnan.(stats_values))
            @test isnan.(stats_values) == isnan.(read_optimizer_stats(outputs)[!, n])
        else
            @test stats_values == read_optimizer_stats(outputs)[!, n]
        end
    end

    for i in 1:executions
        @test get_objective_value(outputs, i) isa Float64
    end
end

@testset "Run EmulationModel with auto-build" begin
    for serialize in (true, false)
        template = get_thermal_dispatch_template_network()
        c_sys5 = PSB.build_system(
            PSITestSystems,
            "c_sys5_uc";
            add_single_time_series = true,
            force_build = true,
        )

        model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
        @test_throws ErrorException run!(model, executions = 10)
        @test run!(
            model;
            executions = 10,
            output_dir = mktempdir(; cleanup = true),
            export_optimization_model = serialize,
        ) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
    end
end

@testset "Test serialization/deserialization of EmulationModel outputs" begin
    path = mktempdir(; cleanup = true)
    template = get_thermal_dispatch_template_network()
    c_sys5 = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc";
        add_single_time_series = true,
        force_build = true,
    )

    model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
    executions = 10
    @test build!(model; executions = executions, output_dir = path) ==
          PSI.ModelBuildStatus.BUILT
    @test run!(model; export_problem_outputs = true) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
    outputs1 = OptimizationProblemOutputs(model)
    var1_a = read_variable(outputs1, ActivePowerVariable, ThermalStandard)
    # Ensure that we can deserialize strings into keys.
    var1_b = read_variable(outputs1, "ActivePowerVariable__ThermalStandard")
    @test var1_a == var1_b

    # Outputs were automatically serialized here.
    outputs2 = OptimizationProblemOutputs(PSI.get_output_dir(model))
    var2 = read_variable(outputs2, ActivePowerVariable, ThermalStandard)
    @test var1_a == var2
    @test get_system(outputs2) === nothing
    get_system!(outputs2)
    @test get_system(outputs2) isa PSY.System

    # Serialize to a new directory with the exported function.
    outputs_path = joinpath(path, "outputs")
    serialize_outputs(outputs1, outputs_path)
    @test isfile(joinpath(outputs_path, ISOPT._PROBLEM_OUTPUTS_FILENAME))
    outputs3 = OptimizationProblemOutputs(outputs_path)
    var3 = read_variable(outputs3, ActivePowerVariable, ThermalStandard)
    @test var1_a == var3
    @test get_system(outputs3) === nothing
    set_system!(outputs3, get_system(outputs1))
    @test get_system(outputs3) !== nothing

    exp_file =
        joinpath(path, "outputs", "variables", "ActivePowerVariable__ThermalStandard.csv")
    var4 = read_dataframe(exp_file)
    # Manually Multiply by the base power var1_a has natural units and export writes directly from the solver
    @test var1_a.value == var4.value .* 100.0
end

@testset "Test deserialization and re-run of EmulationModel" begin
    path = mktempdir(; cleanup = true)
    template = get_thermal_dispatch_template_network()
    c_sys5 = PSB.build_system(
        PSITestSystems,
        "c_sys5_uc";
        add_single_time_series = true,
        force_build = true,
    )

    model = EmulationModel(template, c_sys5; optimizer = HiGHS_optimizer)
    executions = 10
    @test build!(model; executions = executions, output_dir = path) ==
          PSI.ModelBuildStatus.BUILT
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
    outputs = OptimizationProblemOutputs(model)
    var1 = read_variable(outputs, ActivePowerVariable, ThermalStandard)

    file_list = sort!(collect(readdir(path)))
    @test PSI._JUMP_MODEL_FILENAME in file_list
    @test PSI._SERIALIZED_MODEL_FILENAME in file_list
    path2 = joinpath(path, "tmp")
    model2 = EmulationModel(path, HiGHS_optimizer)
    build!(model2; output_dir = path2)
    @test run!(model2) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
    outputs2 = OptimizationProblemOutputs(model2)
    var2 = read_variable(outputs, ActivePowerVariable, ThermalStandard)

    @test var1 == var2

    # Deserialize with different optimizer attributes.
    optimizer = JuMP.optimizer_with_attributes(HiGHS.Optimizer, "time_limit" => 110.0)
    @test_logs (:warn, r"Original solver was .*, new solver is") match_mode = :any EmulationModel(
        path,
        optimizer,
    )

    # Deserialize with a different optimizer.
    @test_logs (:warn, r"Original solver was .* new solver is") match_mode = :any EmulationModel(
        path,
        HiGHS_optimizer,
    )
end

@testset "Test serialization of InitialConditionsData" begin
    template = get_thermal_standard_uc_template()
    sys = PSB.build_system(
        PSITestSystems,
        "c_sys5_pglib";
        add_single_time_series = true,
        force_build = true,
    )
    optimizer = HiGHS_optimizer
    set_device_model!(template, ThermalMultiStart, ThermalMultiStartUnitCommitment)
    model = EmulationModel(template, sys; optimizer = HiGHS_optimizer)
    output_dir = mktempdir(; cleanup = true)

    @test build!(model; executions = 1, output_dir = output_dir) ==
          PSI.ModelBuildStatus.BUILT
    ic_file = PSI.get_initial_conditions_file(model)
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "make")
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    # Build again, use existing initial conditions.
    PSI.reset!(model)
    @test build!(model; executions = 1, output_dir = output_dir) ==
          PSI.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "make")
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    # Build again, use existing initial conditions.
    model = EmulationModel(
        template,
        sys;
        optimizer = optimizer,
        deserialize_initial_conditions = true,
    )
    @test build!(model; executions = 1, output_dir = output_dir) ==
          PSI.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "deserialize")
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED

    # Construct and build again with custom initial conditions file.
    initialization_file = joinpath(output_dir, ic_file * ".old")
    mv(ic_file, initialization_file)
    touch(ic_file)
    model = EmulationModel(
        template,
        sys;
        optimizer = optimizer,
        initialization_file = initialization_file,
        deserialize_initial_conditions = true,
    )
    @test build!(model; executions = 1, output_dir = output_dir) ==
          PSI.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = true, message = "deserialize")

    # Construct and build again while skipping build of initial conditions.
    model = EmulationModel(template, sys; optimizer = optimizer, initialize_model = false)
    rm(ic_file)
    @test build!(model; executions = 1, output_dir = output_dir) ==
          PSI.ModelBuildStatus.BUILT
    test_ic_serialization_outputs(model; ic_file_exists = false, message = "skip")
    @test run!(model) == PSI.RunStatus.SUCCESSFULLY_FINALIZED
end
