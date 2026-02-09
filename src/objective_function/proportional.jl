# add_proportional_cost! is used for Thermals (a bunch) and ControllableLoads (once) in POM
# add_proportional_cost_maybe_time_variant! is used to define a add_proportional_cost! in
# POM, for Thermals and ControllableLoads with certain formulations.

# this is only used for ControllableLoads with non-PowerLoadInterruptible formulations.
# The rest go through a thin wrapper around the maybe-variant version.
"""
Default implementation for proportional cost, where the cost term is not time variant. Anything
time-varying should implement its own method.
"""
function add_proportional_cost!(
    container::OptimizationContainer,
    ::U,
    devices::IS.FlattenIteratorWrapper{T},
    ::V,
) where {
    T <: IS.InfrastructureSystemsComponent,
    U <: VariableType,
    V <: AbstractDeviceFormulation,
}
    # NOTE: anything time-varying should implement its own method.
    multiplier = objective_function_multiplier(U(), V())
    for d in devices
        op_cost_data = get_operation_cost(d)
        cost_term = proportional_cost(op_cost_data, U(), d, V())
        iszero(cost_term) && continue
        name = get_name(d)
        rate = cost_term * multiplier
        for t in get_time_steps(container)
            variable = get_variable(container, U(), T)[name, t]
            add_cost_term_invariant!(
                container,
                variable,
                rate,
                ProductionCostExpression,
                T,
                name,
                t,
            )
        end
    end
    return
end

# corresponds to get_must_run for thermals, but avoiding device specific code here.
"""
Extension point: whether to skip adding proportional cost for a given device.
For thermals, equivalent to `get_must_run`, but that implementation belongs in POM.
"""
skip_proportional_cost(d::IS.InfrastructureSystemsComponent) = false

"""
Common basis for maybe time variant proportional costs for devices that might have must-run behavior.
Currently used for `(ThermalGen, AbstractThermal)` and `(ControllableLoad, PowerLoadInterruption)`
device, formulation pairs.
"""
function add_proportional_cost_maybe_time_variant!(
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
        op_cost_data = get_operation_cost(d)
        name = get_name(d)
        for t in get_time_steps(container)
            cost_term = proportional_cost(container, op_cost_data, U(), d, V(), t)
            iszero(cost_term) && continue
            rate = cost_term * multiplier

            if skip_proportional_cost(d)
                # Only add to expression, not objective
                add_cost_to_expression!(
                    container,
                    ProductionCostExpression,
                    rate,
                    T,
                    name,
                    t,
                )
            else
                variable = get_variable(container, U(), T)[name, t]
                add_as_time_variant =
                    is_time_variant_term(container, op_cost_data, U(), T, V(), t)
                if add_as_time_variant
                    add_cost_term_variant!(
                        container, variable, rate, ProductionCostExpression, T, name, t)
                else
                    add_cost_term_invariant!(
                        container, variable, rate, ProductionCostExpression, T, name, t)
                end
            end
        end
    end
    return
end
