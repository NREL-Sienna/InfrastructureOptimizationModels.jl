function _setup_qa_container(time_steps::UnitRange{Int})
    sys = MockSystem(100.0)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
    )
    container = IOM.OptimizationContainer(sys, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, time_steps)
    return container
end

function _setup_qa_test(device_names::Vector{String}, time_steps::UnitRange{Int})
    container = _setup_qa_container(time_steps)
    var_container = IOM.add_variable_container!(
        container,
        TestOriginalVariable(),
        MockThermalGen,
        device_names,
        time_steps,
    )
    jump_model = IOM.get_jump_model(container)
    for name in device_names, t in time_steps
        var_container[name, t] = JuMP.@variable(jump_model, base_name = "x_$(name)_$(t)",)
    end
    return (; container, var_container, jump_model)
end

function _setup_bilinear_test(device_names::Vector{String}, time_steps::UnitRange{Int})
    container = _setup_qa_container(time_steps)
    x_var_container = IOM.add_variable_container!(
        container,
        TestOriginalVariable(),
        MockThermalGen,
        device_names,
        time_steps,
    )
    y_var_container = IOM.add_variable_container!(
        container,
        TestApproximatedVariable(),
        MockThermalGen,
        device_names,
        time_steps,
    )
    jump_model = IOM.get_jump_model(container)
    for name in device_names, t in time_steps
        x_var_container[name, t] =
            JuMP.@variable(jump_model, base_name = "x_$(name)_$(t)")
        y_var_container[name, t] =
            JuMP.@variable(jump_model, base_name = "y_$(name)_$(t)")
    end
    return (; container, x_var_container, y_var_container, jump_model)
end
