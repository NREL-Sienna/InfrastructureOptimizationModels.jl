# ideally would define in POM, but put here for now.
"Parameter to define startup cost time series"
struct StartupCostParameter <: ObjectiveFunctionParameter end

"Parameter to define shutdown cost time series"
struct ShutdownCostParameter <: ObjectiveFunctionParameter end

# Extract the scalar shutdown cost from either a Float64 (ThermalGenerationCost) or
# a LinearCurve (MarketBidCost) whose proportional term is the cost.
_shutdown_cost_value(x::Float64) = x
_shutdown_cost_value(x::IS.LinearCurve) = IS.get_proportional_term(x)

function add_shut_down_cost!(
    container::OptimizationContainer,
    ::U,
    devices::IS.FlattenIteratorWrapper{T},
    ::V,
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    multiplier = objective_function_multiplier(U(), V())
    for d in devices
        get_must_run(d) && continue
        name = get_name(d)
        add_as_time_variant = is_time_variant(get_shut_down(get_operation_cost(d)))
        for t in get_time_steps(container)
            cost_term = if add_as_time_variant
                param = get_parameter_array(container, ShutdownCostParameter, T)
                mult =
                    get_parameter_multiplier_array(container, ShutdownCostParameter, T)
                param[name, t] * mult[name, t]
            else
                _shutdown_cost_value(get_shut_down(get_operation_cost(d)))
            end
            iszero(cost_term) && continue
            rate = cost_term * multiplier
            variable = get_variable(container, U, T)[name, t]
            if add_as_time_variant
                add_cost_term_variant!(
                    container, variable, rate, ProductionCostExpression, T, name, t)
            else
                add_cost_term_invariant!(
                    container, variable, rate, ProductionCostExpression, T, name, t)
            end
        end
    end
    return
end

function add_start_up_cost!(
    container::OptimizationContainer,
    ::U,
    devices::IS.FlattenIteratorWrapper{T},
    ::V,
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    for d in devices
        op_cost_data = get_operation_cost(d)
        _add_start_up_cost_to_objective!(container, U(), d, op_cost_data, V())
    end
    return
end

# op_cost is PSY.ThermalGenerationCost or PSY.MarketBidCost, but trying
# to avoid PSY types here.
function _add_start_up_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::C,
    op_cost,
    ::U,
) where {
    T <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    multiplier = objective_function_multiplier(T(), U())
    get_must_run(component) && return
    name = get_name(component)
    add_as_time_variant = is_time_variant(get_start_up(op_cost))
    for t in get_time_steps(container)
        cost_term = get_startup_cost_value(
            container, T(), component, U(), t, add_as_time_variant)
        iszero(cost_term) && continue
        rate = cost_term * multiplier
        variable = get_variable(container, T, C)[name, t]
        if add_as_time_variant
            add_cost_term_variant!(
                container, variable, rate, ProductionCostExpression, C, name, t)
        else
            add_cost_term_invariant!(
                container, variable, rate, ProductionCostExpression, C, name, t)
        end
    end
    return
end

function get_startup_cost_value(
    container::OptimizationContainer,
    ::T,
    component::V,
    ::U,
    time_period::Int,
    is_time_variant_::Bool,
) where {
    T <: VariableType,
    V <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    raw_startup_cost = if is_time_variant_
        name = get_name(component)
        param = get_parameter_array(container, StartupCostParameter, V)
        mult = get_parameter_multiplier_array(container, StartupCostParameter, V)
        param[name, time_period] * mult[name, time_period]
    else
        get_start_up(get_operation_cost(component))
    end
    # possible types for raw_startup_cost: Float, AffExpr, NamedTuple, or StartUpStages.
    return start_up_cost(raw_startup_cost, V, T(), U())
end
