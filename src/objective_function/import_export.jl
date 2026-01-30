# FIXME requires AbstractSourceFormulation to be defined
# or, rely on PSY.Source being enough to uniquely determine which function gets called.

#=
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::PSY.Source,
    cost_function::PSY.ImportExportCost,
    ::U,
) where {
    T <: ActivePowerOutVariable,
    U <: AbstractSourceFormulation,
}
    component_name = PSY.get_name(component)
    @debug "Import Export Cost" _group = LOG_GROUP_COST_FUNCTIONS component_name
    import_cost_curves = PSY.get_import_offer_curves(cost_function)
    if !isnothing(import_cost_curves)
        add_pwl_term!(
            false,
            container,
            component,
            cost_function,
            T(),
            U(),
        )
    end
    return
end

function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::PSY.Source,
    cost_function::PSY.ImportExportCost,
    ::U,
) where {
    T <: ActivePowerInVariable,
    U <: AbstractSourceFormulation,
}
    component_name = PSY.get_name(component)
    @debug "Import Export Cost" _group = LOG_GROUP_COST_FUNCTIONS component_name
    export_cost_curves = PSY.get_export_offer_curves(cost_function)
    if !isnothing(export_cost_curves)
        add_pwl_term!(
            true,
            container,
            component,
            cost_function,
            T(),
            U(),
        )
    end
    return
end
=#
