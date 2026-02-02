# ideally would define in POM, but put here for now.
"Parameter to define startup cost time series"
struct StartupCostParameter <: ObjectiveFunctionParameter end

"Parameter to define shutdown cost time series"
struct ShutdownCostParameter <: ObjectiveFunctionParameter end

get_shutdown_cost_value(
    container::OptimizationContainer,
    component::IS.InfrastructureSystemsComponent,
    time_period::Int,
    is_time_variant_::Bool,
) = _lookup_maybe_time_variant_param(
    container,
    component,
    time_period,
    Val(is_time_variant_),
    get_shut_down ∘ get_operation_cost,
    ShutdownCostParameter(),
)

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

        add_as_time_variant = is_time_variant(get_shut_down(get_operation_cost(d)))
        for t in get_time_steps(container)
            my_cost_term = get_shutdown_cost_value(
                container,
                d,
                t,
                add_as_time_variant,
            )
            iszero(my_cost_term) && continue
            exp = _add_proportional_term_maybe_variant!(
                Val(add_as_time_variant), container, U(), d, my_cost_term * multiplier,
                t)
            add_cost_to_expression!(container, ProductionCostExpression, exp, d, t)
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

# NOTE: Type constraints PSY.ThermalGen and PSY.{ThermalGenerationCost, MarketBidCost}
# are device/cost-specific and should eventually move to POM.
# Alternative: replace with any component and any operation cost, then write thin wrapers in POM.
function _add_start_up_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::PSY.ThermalGen,
    op_cost::Union{PSY.ThermalGenerationCost, PSY.MarketBidCost},
    ::U,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    multiplier = objective_function_multiplier(T(), U())
    get_must_run(component) && return
    add_as_time_variant = is_time_variant(get_start_up(op_cost))
    for t in get_time_steps(container)
        my_cost_term = get_startup_cost_value(
            container,
            T(),
            component,
            U(),
            t,
            add_as_time_variant,
        )
        iszero(my_cost_term) && continue
        exp = _add_proportional_term_maybe_variant!(
            Val(add_as_time_variant), container, T(), component,
            my_cost_term * multiplier, t)
        add_cost_to_expression!(container, ProductionCostExpression, exp, component, t)
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
    raw_startup_cost = _lookup_maybe_time_variant_param(
        container,
        component,
        time_period,
        Val(is_time_variant_),
        get_start_up ∘ get_operation_cost,
        StartupCostParameter(),
    )
    # TODO add stub for start_up_cost.
    return start_up_cost(raw_startup_cost, component, T(), U())
end
