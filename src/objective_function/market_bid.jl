#################################################################################
# Market Bid Cost / Import-Export Cost: Generic Optimization Infrastructure
#
# This file provides the generic (device-agnostic) infrastructure for
# MarketBidCost and ImportExportCost offer curves. Device-specific overloads
# (e.g., for ThermalMultiStart, ControllableLoad formulations) are in POM's
# market_bid_overrides.jl.
#################################################################################

#################################################################################
# Section 1: Offer Curve Accessor Wrappers
# Map MarketBidCost / ImportExportCost to a unified interface.
#################################################################################

get_output_offer_curves(cost::PSY.ImportExportCost, args...; kwargs...) =
    PSY.get_import_offer_curves(cost, args...; kwargs...)
get_output_offer_curves(cost::PSY.MarketBidCost, args...; kwargs...) =
    PSY.get_incremental_offer_curves(cost, args...; kwargs...)
get_input_offer_curves(cost::PSY.ImportExportCost, args...; kwargs...) =
    PSY.get_export_offer_curves(cost, args...; kwargs...)
get_input_offer_curves(cost::PSY.MarketBidCost, args...; kwargs...) =
    PSY.get_decremental_offer_curves(cost, args...; kwargs...)

get_output_offer_curves(
    component::PSY.Component,
    cost::PSY.ImportExportCost,
    args...;
    kwargs...,
) =
    PSY.get_import_offer_curves(component, cost, args...; kwargs...)
get_output_offer_curves(
    component::PSY.Component,
    cost::PSY.MarketBidCost,
    args...;
    kwargs...,
) =
    PSY.get_incremental_offer_curves(component, cost, args...; kwargs...)
get_input_offer_curves(
    component::PSY.Component,
    cost::PSY.ImportExportCost,
    args...;
    kwargs...,
) =
    PSY.get_export_offer_curves(component, cost, args...; kwargs...)
get_input_offer_curves(
    component::PSY.Component,
    cost::PSY.MarketBidCost,
    args...;
    kwargs...,
) =
    PSY.get_decremental_offer_curves(component, cost, args...; kwargs...)

# OfferDirection-based accessors
get_initial_input(::DecrementalOffer, device::PSY.StaticInjection) =
    PSY.get_decremental_initial_input(PSY.get_operation_cost(device))
get_initial_input(::IncrementalOffer, device::PSY.StaticInjection) =
    PSY.get_incremental_initial_input(PSY.get_operation_cost(device))

get_offer_curves(::DecrementalOffer, device::PSY.StaticInjection) =
    get_input_offer_curves(PSY.get_operation_cost(device))
get_offer_curves(::IncrementalOffer, device::PSY.StaticInjection) =
    get_output_offer_curves(PSY.get_operation_cost(device))

# Overloads that accept cost object directly (used by VOM cost path)
get_offer_curves(::DecrementalOffer, op_cost::PSY.OfferCurveCost) =
    get_input_offer_curves(op_cost)
get_offer_curves(::IncrementalOffer, op_cost::PSY.OfferCurveCost) =
    get_output_offer_curves(op_cost)

#################################################################################
# Section 2: OfferDirection Type Dispatch Table
# Maps OfferDirection to the appropriate parameter/variable/constraint types.
#################################################################################

_slope_param(::IncrementalOffer) = IncrementalPiecewiseLinearSlopeParameter
_slope_param(::DecrementalOffer) = DecrementalPiecewiseLinearSlopeParameter

_breakpoint_param(::IncrementalOffer) = IncrementalPiecewiseLinearBreakpointParameter
_breakpoint_param(::DecrementalOffer) = DecrementalPiecewiseLinearBreakpointParameter

_block_offer_var(::IncrementalOffer) = PiecewiseLinearBlockIncrementalOffer
_block_offer_var(::DecrementalOffer) = PiecewiseLinearBlockDecrementalOffer

_block_offer_constraint(::IncrementalOffer) = PiecewiseLinearBlockIncrementalOfferConstraint
_block_offer_constraint(::DecrementalOffer) = PiecewiseLinearBlockDecrementalOfferConstraint

_objective_sign(::IncrementalOffer) = OBJECTIVE_FUNCTION_POSITIVE
_objective_sign(::DecrementalOffer) = OBJECTIVE_FUNCTION_NEGATIVE

#################################################################################
# Section 3: _get_parameter_field Dispatch Table
# Maps parameter types to PSY getter functions.
#################################################################################

_get_parameter_field(::StartupCostParameter, args...; kwargs...) =
    PSY.get_start_up(args...; kwargs...)
_get_parameter_field(::ShutdownCostParameter, args...; kwargs...) =
    PSY.get_shut_down(args...; kwargs...)
_get_parameter_field(::IncrementalCostAtMinParameter, args...; kwargs...) =
    PSY.get_incremental_initial_input(args...; kwargs...)
_get_parameter_field(::DecrementalCostAtMinParameter, args...; kwargs...) =
    PSY.get_decremental_initial_input(args...; kwargs...)
_get_parameter_field(
    ::Union{
        IncrementalPiecewiseLinearSlopeParameter,
        IncrementalPiecewiseLinearBreakpointParameter,
    },
    args...;
    kwargs...,
) =
    get_output_offer_curves(args...; kwargs...)
_get_parameter_field(
    ::Union{
        DecrementalPiecewiseLinearSlopeParameter,
        DecrementalPiecewiseLinearBreakpointParameter,
    },
    args...;
    kwargs...,
) =
    get_input_offer_curves(args...; kwargs...)

#################################################################################
# Section 4: Device Cost Detection Predicates (generic)
# Device-specific overrides (RenewableNonDispatch, PowerLoad, etc.) are in POM.
#################################################################################

_has_market_bid_cost(device::PSY.StaticInjection) =
    PSY.get_operation_cost(device) isa PSY.MarketBidCost

_has_import_export_cost(device::PSY.Source) =
    PSY.get_operation_cost(device) isa PSY.ImportExportCost
_has_import_export_cost(::PSY.StaticInjection) = false

_has_offer_curve_cost(device::PSY.Component) =
    _has_market_bid_cost(device) || _has_import_export_cost(device)

_has_parameter_time_series(::StartupCostParameter, device::PSY.StaticInjection) =
    is_time_variant(PSY.get_start_up(PSY.get_operation_cost(device)))

_has_parameter_time_series(::ShutdownCostParameter, device::PSY.StaticInjection) =
    is_time_variant(PSY.get_shut_down(PSY.get_operation_cost(device)))

_has_parameter_time_series(
    ::T,
    device::PSY.StaticInjection,
) where {T <: AbstractCostAtMinParameter} =
    _has_offer_curve_cost(device) &&
    is_time_variant(_get_parameter_field(T(), PSY.get_operation_cost(device)))

_has_parameter_time_series(
    ::T,
    device::PSY.StaticInjection,
) where {T <: AbstractPiecewiseLinearSlopeParameter} =
    _has_offer_curve_cost(device) &&
    is_time_variant(_get_parameter_field(T(), PSY.get_operation_cost(device)))

_has_parameter_time_series(
    ::T,
    device::PSY.StaticInjection,
) where {T <: AbstractPiecewiseLinearBreakpointParameter} =
    _has_offer_curve_cost(device) &&
    is_time_variant(_get_parameter_field(T(), PSY.get_operation_cost(device)))

#################################################################################
# Section 5: _consider_parameter (generic versions)
# Whether a parameter should be added based on what's in the container.
# POM overrides for ThermalMultiStart startup (MULTI_START_VARIABLES).
#################################################################################

_consider_parameter(
    ::StartupCostParameter,
    container::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = has_container_key(container, StartVariable, T)

_consider_parameter(
    ::ShutdownCostParameter,
    container::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = has_container_key(container, StopVariable, T)

_consider_parameter(
    ::AbstractCostAtMinParameter,
    container::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = has_container_key(container, OnVariable, T)

_consider_parameter(
    ::AbstractPiecewiseLinearSlopeParameter,
    ::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = true

_consider_parameter(
    ::AbstractPiecewiseLinearBreakpointParameter,
    ::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = true

#################################################################################
# Section 6: Validation
# Generic validation for offer curve costs. Device-specific overrides
# (ThermalMultiStart, RenewableDispatch, Storage) are in POM.
#################################################################################

function validate_initial_input_time_series(
    device::PSY.StaticInjection,
    dir::OfferDirection,
)
    initial_input = get_initial_input(dir, device)
    initial_is_ts = is_time_variant(initial_input)
    variable_is_ts = is_time_variant(get_offer_curves(dir, device))
    label = dir isa DecrementalOffer ? "decremental" : "incremental"

    (initial_is_ts && !variable_is_ts) &&
        @warn "In `MarketBidCost` for $(get_name(device)), found time series for `$(label)_initial_input` but non-time-series `$(label)_offer_curves`; will ignore `initial_input` of `$(label)_offer_curves"
    (variable_is_ts && !initial_is_ts) &&
        throw(
            ArgumentError(
                "In `MarketBidCost` for $(get_name(device)), if providing time series for `$(label)_offer_curves`, must also provide time series for `$(label)_initial_input`",
            ),
        )

    if !variable_is_ts && !initial_is_ts
        _validate_eltype(
            Union{Float64, Nothing}, device, initial_input, " initial_input",
        )
    else
        _validate_eltype(
            Float64, device, initial_input, " initial_input",
        )
    end
end

function validate_occ_breakpoints_slopes(device::PSY.StaticInjection, dir::OfferDirection)
    offer_curves = get_offer_curves(dir, device)
    device_name = get_name(device)
    is_ts = is_time_variant(offer_curves)
    expected_type = if is_ts
        IS.PiecewiseStepData
    else
        PSY.CostCurve{PSY.PiecewiseIncrementalCurve}
    end
    p1 = nothing
    apply_maybe_across_time_series(device, offer_curves) do x
        curve_type = dir isa DecrementalOffer ? "decremental" : "incremental"
        _validate_eltype(expected_type, device, x, " $curve_type offer curves")
        if dir isa DecrementalOffer
            PSY.is_concave(x) ||
                throw(
                    ArgumentError(
                        "Decremental $(nameof(typeof(PSY.get_operation_cost(device)))) for component $(device_name) is non-concave",
                    ),
                )
        else
            PSY.is_convex(x) ||
                throw(
                    ArgumentError(
                        "Incremental $(nameof(typeof(PSY.get_operation_cost(device)))) for component $(device_name) is non-convex",
                    ),
                )
        end

        p1 = _validate_occ_subtype(
            PSY.get_operation_cost(device),
            dir,
            is_ts,
            x,
            device_name,
            p1,
        )
    end
end

function _validate_occ_subtype(
    ::PSY.MarketBidCost,
    dir::OfferDirection,
    is_ts,
    curve::PSY.PiecewiseStepData,
    device_name::String,
    p1::Union{Nothing, Float64},
)
    @assert is_ts
    my_p1 = first(PSY.get_x_coords(curve))
    if isnothing(p1)
        p1 = my_p1
    elseif !isapprox(p1, my_p1)
        throw(
            ArgumentError(
                "Inconsistent minimum breakpoint values in time series MarketBidCost for $(device_name) offer curves. For time-variable MarketBidCost, all first x-coordinates must be equal across the entire time series.",
            ),
        )
    end
    return p1
end

_validate_occ_subtype(
    ::PSY.MarketBidCost,
    dir::OfferDirection,
    is_ts,
    ::PSY.CostCurve,
    args...,
) =
    @assert !is_ts

function _validate_occ_subtype(
    cost::PSY.ImportExportCost,
    dir::OfferDirection,
    is_ts,
    curve::PSY.CostCurve,
    args...,
)
    @assert !is_ts
    !iszero(PSY.get_vom_cost(curve)) && throw(
        ArgumentError(
            "For ImportExportCost, VOM cost must be zero.",
        ),
    )
    vc = PSY.get_value_curve(curve)
    !iszero(PSY.get_initial_input(curve)) && throw(
        ArgumentError(
            "For ImportExportCost, initial input must be zero.",
        ),
    )
    _validate_occ_subtype(cost, dir, true, PSY.get_function_data(vc))
end

function _validate_occ_subtype(
    ::PSY.ImportExportCost,
    dir::OfferDirection,
    is_ts,
    curve::PSY.PiecewiseStepData,
    args...,
)
    @assert is_ts
    if !iszero(first(PSY.get_x_coords(curve)))
        throw(
            ArgumentError(
                "For ImportExportCost, the first breakpoint must be zero.",
            ),
        )
    end
end

# Generic validate_occ_component overloads for PSY.StaticInjection.
# Device-specific overloads (ThermalMultiStart, RenewableDispatch, Storage) are in POM.

function validate_occ_component(::StartupCostParameter, device::PSY.StaticInjection)
    startup = PSY.get_start_up(PSY.get_operation_cost(device))
    contains_multistart = false
    apply_maybe_across_time_series(device, startup) do x
        if x isa Float64
            return
        elseif x isa Union{NTuple{3, Float64}, StartUpStages}
            contains_multistart = true
        else
            location =
                is_time_variant(startup) ? " in time series $(get_name(startup))" : ""
            throw(
                ArgumentError(
                    "Expected Float64 or NTuple{3, Float64} or StartUpStages startup cost but got $(typeof(x))$location for $(get_name(device))",
                ),
            )
        end
    end
    if contains_multistart
        location = is_time_variant(startup) ? " in time series $(get_name(startup))" : ""
        @warn "Multi-start costs detected$location for non-multi-start unit $(get_name(device)), will take the maximum"
    end
    return
end

function validate_occ_component(::ShutdownCostParameter, device::PSY.StaticInjection)
    shutdown = PSY.get_shut_down(PSY.get_operation_cost(device))
    _validate_eltype(Float64, device, shutdown, " for shutdown cost")
end

validate_occ_component(
    ::IncrementalCostAtMinParameter,
    device::PSY.StaticInjection,
) = validate_initial_input_time_series(device, IncrementalOffer())

validate_occ_component(
    ::DecrementalCostAtMinParameter,
    device::PSY.StaticInjection,
) = validate_initial_input_time_series(device, DecrementalOffer())

validate_occ_component(
    ::IncrementalPiecewiseLinearBreakpointParameter,
    device::PSY.StaticInjection,
) = validate_occ_breakpoints_slopes(device, IncrementalOffer())

validate_occ_component(
    ::DecrementalPiecewiseLinearBreakpointParameter,
    device::PSY.StaticInjection,
) = validate_occ_breakpoints_slopes(device, DecrementalOffer())

# Slope and breakpoint validations are done together, nothing to do here
validate_occ_component(
    ::AbstractPiecewiseLinearSlopeParameter,
    device::PSY.StaticInjection,
) = nothing

#################################################################################
# Section 7: Parameter Processing Orchestration
#################################################################################

function _process_occ_parameters_helper(
    ::P,
    container::OptimizationContainer,
    model,
    devices,
) where {P <: ParameterType}
    param_instance = P()
    for device in devices
        validate_occ_component(param_instance, device)
    end
    if _consider_parameter(param_instance, container, model)
        ts_devices =
            filter(device -> _has_parameter_time_series(param_instance, device), devices)
        (length(ts_devices) > 0) && add_parameters!(container, P, ts_devices, model)
    end
end

"Validate ImportExportCosts and add the appropriate parameters"
function process_import_export_parameters!(
    container::OptimizationContainer,
    devices_in,
    model::DeviceModel,
)
    devices = filter(_has_import_export_cost, collect(devices_in))

    for param in (
        IncrementalPiecewiseLinearSlopeParameter(),
        IncrementalPiecewiseLinearBreakpointParameter(),
        DecrementalPiecewiseLinearSlopeParameter(),
        DecrementalPiecewiseLinearBreakpointParameter(),
    )
        _process_occ_parameters_helper(param, container, model, devices)
    end
end

"Validate MarketBidCosts and add the appropriate parameters"
function process_market_bid_parameters!(
    container::OptimizationContainer,
    devices_in,
    model::DeviceModel,
    incremental::Bool = true,
    decremental::Bool = false,
)
    devices = filter(_has_market_bid_cost, collect(devices_in))
    isempty(devices) && return

    for param in (
        StartupCostParameter(),
        ShutdownCostParameter(),
    )
        _process_occ_parameters_helper(param, container, model, devices)
    end
    if incremental
        for param in (
            IncrementalCostAtMinParameter(),
            IncrementalPiecewiseLinearSlopeParameter(),
            IncrementalPiecewiseLinearBreakpointParameter(),
        )
            _process_occ_parameters_helper(param, container, model, devices)
        end
    end
    if decremental
        for param in (
            DecrementalCostAtMinParameter(),
            DecrementalPiecewiseLinearSlopeParameter(),
            DecrementalPiecewiseLinearBreakpointParameter(),
        )
            _process_occ_parameters_helper(param, container, model, devices)
        end
    end
end

#################################################################################
# Section 8: Min-Gen-Power Dispatch Defaults
# POM overrides these for specific device types and formulations.
#################################################################################

_include_min_gen_power_in_constraint(
    ::Type,
    ::VariableType,
    ::AbstractDeviceFormulation,
) = false

_include_constant_min_gen_power_in_constraint(
    ::Type,
    ::VariableType,
    ::AbstractDeviceFormulation,
) = false

#################################################################################
# Section 9: PWL Block Offer Constraints (generic)
# The ReserveDemandCurve-specific overload is in POM.
#################################################################################

"""
Implement the constraints for PWL Block Offer variables. That is:

```math
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} = p_t \\\\
\\sum_{k\\in\\mathcal{K}} \\delta_{k,t} <= P_{k+1,t}^{max} - P_{k,t}^{max}
```
"""
function _add_pwl_constraint!(
    container::OptimizationContainer,
    component::T,
    ::U,
    ::D,
    break_points::Vector{<:JuMPOrFloat},
    pwl_vars::Vector{JuMP.VariableRef},
    period::Int,
    ::Type{W},
) where {T <: PSY.Component, U <: VariableType,
    D <: AbstractDeviceFormulation,
    W <: AbstractPiecewiseLinearBlockOfferConstraint}
    variables = get_variable(container, U(), T)
    const_container = lazy_container_addition!(
        container,
        W(),
        T,
        axes(variables)...,
    )
    name = PSY.get_name(component)

    min_power_offset = if _include_constant_min_gen_power_in_constraint(T, U(), D())
        jump_fixed_value(first(break_points))::Float64
    elseif _include_min_gen_power_in_constraint(T, U(), D())
        p1::Float64 = jump_fixed_value(first(break_points))
        on_vars = get_variable(container, OnVariable(), T)
        p1 * on_vars[name, period]
    else
        0.0
    end

    add_pwl_block_offer_constraints!(
        get_jump_model(container),
        const_container,
        name,
        period,
        variables[name, period],
        pwl_vars,
        break_points,
        min_power_offset,
    )
    return
end

#################################################################################
# Section 10: PWL Data Retrieval
#################################################################################

function _get_pwl_data(
    dir::OfferDirection,
    container::OptimizationContainer,
    component::T,
    time::Int,
) where {T <: PSY.Component}
    cost_data = get_offer_curves(dir, component)

    if is_time_variant(cost_data)
        name = PSY.get_name(component)

        SlopeParam = _slope_param(dir)
        slope_param_arr = get_parameter_array(container, SlopeParam(), T)
        slope_param_mult = get_parameter_multiplier_array(container, SlopeParam(), T)
        @assert size(slope_param_arr) == size(slope_param_mult)
        slope_cost_component =
            slope_param_arr[name, :, time] .* slope_param_mult[name, :, time]
        slope_cost_component = slope_cost_component.data

        BreakpointParam = _breakpoint_param(dir)
        breakpoint_param_container = get_parameter(container, BreakpointParam(), T)
        breakpoint_param_arr = get_parameter_column_refs(breakpoint_param_container, name)
        breakpoint_param_mult = get_multiplier_array(breakpoint_param_container)
        @assert size(breakpoint_param_arr) == size(breakpoint_param_mult[name, :, :])
        breakpoint_cost_component =
            breakpoint_param_arr[:, time] .* breakpoint_param_mult[name, :, time]
        breakpoint_cost_component = breakpoint_cost_component.data

        @assert_op length(slope_cost_component) == length(breakpoint_cost_component) - 1
        unit_system = PSY.UnitSystem.NATURAL_UNITS
    else
        cost_component = PSY.get_function_data(PSY.get_value_curve(cost_data))
        breakpoint_cost_component = PSY.get_x_coords(cost_component)
        slope_cost_component = PSY.get_y_coords(cost_component)
        unit_system = PSY.get_power_units(cost_data)
    end

    breakpoints, slopes = get_piecewise_curve_per_system_unit(
        breakpoint_cost_component,
        slope_cost_component,
        unit_system,
        get_model_base_power(container),
        PSY.get_base_power(component),
    )

    return breakpoints, slopes
end

#################################################################################
# Section 11: PWL Cost Terms + Variable Cost Objective (generic)
# Load formulation overloads (AbstractControllablePowerLoadFormulation) are in POM.
#################################################################################

"""
Add PWL cost terms for data coming from a MarketBidCost / ImportExportCost
with offer curves, dispatched on OfferDirection.
"""
function add_pwl_term!(
    dir::OfferDirection,
    container::OptimizationContainer,
    component::T,
    ::PSY.OfferCurveCost,
    ::U,
    ::V,
) where {T <: PSY.Component, U <: VariableType, V <: AbstractDeviceFormulation}
    W = _block_offer_var(dir)
    X = _block_offer_constraint(dir)

    name = PSY.get_name(component)
    resolution = get_resolution(container)
    dt = Dates.value(resolution) / MILLISECONDS_IN_HOUR
    time_steps = get_time_steps(container)
    for t in time_steps
        breakpoints, slopes = _get_pwl_data(dir, container, component, t)
        pwl_vars =
            add_pwl_variables!(container, W, T, name, t, length(slopes); upper_bound = Inf)
        _add_pwl_constraint!(
            container,
            component,
            U(),
            V(),
            breakpoints,
            pwl_vars,
            t,
            X,
        )
        pwl_cost = get_pwl_cost_expression(pwl_vars, slopes, _objective_sign(dir) * dt)

        add_cost_to_expression!(
            container,
            ProductionCostExpression,
            pwl_cost,
            T,
            name,
            t,
        )

        if is_time_variant(get_offer_curves(dir, component))
            add_to_objective_variant_expression!(container, pwl_cost)
        else
            add_to_objective_invariant_expression!(container, pwl_cost)
        end
    end
end

"""
Generic: incremental offers only (most device formulations).
Decremental-only overload for load formulations is in POM.
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::PSY.Component,
    cost_function::PSY.OfferCurveCost,
    ::U,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    component_name = PSY.get_name(component)
    @debug "Market Bid" _group = LOG_GROUP_COST_FUNCTIONS component_name
    if !isnothing(get_input_offer_curves(cost_function))
        error("Component $(component_name) is not allowed to participate as a demand.")
    end
    add_pwl_term!(
        IncrementalOffer(),
        container,
        component,
        cost_function,
        T(),
        U(),
    )
    return
end

# Default: most formulations use incremental offers
_vom_offer_direction(::AbstractDeviceFormulation) = IncrementalOffer()

function _add_vom_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::PSY.Component,
    op_cost::PSY.OfferCurveCost,
    ::U,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    dir = _vom_offer_direction(U())
    cost_curves = get_offer_curves(dir, op_cost)
    if is_time_variant(cost_curves)
        @warn "$(typeof(dir)) curves are time variant, there is no VOM cost source. Skipping VOM cost."
        return
    end
    _add_vom_cost_to_objective_helper!(
        container, T(), component, op_cost, cost_curves, U())
    return
end

function _add_vom_cost_to_objective_helper!(
    container::OptimizationContainer,
    ::T,
    component::PSY.Component,
    ::PSY.OfferCurveCost,
    cost_data::PSY.CostCurve{PSY.PiecewiseIncrementalCurve},
    ::U,
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    power_units = PSY.get_power_units(cost_data)
    cost_term = PSY.get_proportional_term(PSY.get_vom_cost(cost_data))
    add_proportional_cost_invariant!(container, T, component, cost_term, power_units)
    return
end
