#################################################################################
# Value Curve Objective Function: Delta PWL Formulation
#
# Objective function formulations for ValueCurve-based offer curves using the
# delta (incremental/block) PWL method. Maps ValueCurve types (static and
# time-series-backed) to slopes/breakpoints and routes to the delta formulation
# primitives in objective_function_pwl_delta.jl.
#
# IOM defines objective function formulations — the mathematical structure of
# JuMP objective terms. "Costs" (production cost, fuel cost, etc.) are a
# domain concept defined in POM. This file provides the formulation machinery
# that POM routes specific cost types into. PSY cost types appear in some
# function signatures for dispatch, but the formulations themselves are
# generic over IS.InfrastructureSystemsComponent and IS.ValueCurve types.
#
# Device-specific overloads (e.g., ThermalMultiStart, ControllableLoad) are
# in POM.
#################################################################################

#################################################################################
# Section 1: Offer Curve Accessor Wrappers
# Map PSY cost types (MarketBidCost, ImportExportCost) to a unified interface.
#################################################################################

####################### get_{output/input}_offer_curves #########################
# these 1-argument getters turn into straight getfield calls
get_output_offer_curves(cost::PSY.ImportExportCost) = PSY.get_import_offer_curves(cost)
get_output_offer_curves(cost::PSY.MarketBidCost) = PSY.get_incremental_offer_curves(cost)
get_input_offer_curves(cost::PSY.ImportExportCost) = PSY.get_export_offer_curves(cost)
get_input_offer_curves(cost::PSY.MarketBidCost) = PSY.get_decremental_offer_curves(cost)

# these 2-argument getters return either a TimeArray of curves or a single curve, 
# depending on whether the cost is time varying or not.
get_output_offer_curves(
    component::PSY.Component,
    cost::PSY.ImportExportCost;
    kwargs...,
) = PSY.get_import_offer_curves(component, cost; kwargs...)
get_output_offer_curves(
    component::PSY.Component,
    cost::PSY.MarketBidCost;
    kwargs...,
) = PSY.get_incremental_offer_curves(component, cost; kwargs...)
get_input_offer_curves(
    component::PSY.Component,
    cost::PSY.ImportExportCost;
    kwargs...,
) = PSY.get_export_offer_curves(component, cost; kwargs...)
get_input_offer_curves(
    component::PSY.Component,
    cost::PSY.MarketBidCost;
    kwargs...,
) = PSY.get_decremental_offer_curves(component, cost; kwargs...)

######################### get_offer_curves(direction, ...) ##############################

# direction and device:
get_offer_curves(::DecrementalOffer, device::PSY.StaticInjection) =
    get_input_offer_curves(PSY.get_operation_cost(device))
get_offer_curves(::IncrementalOffer, device::PSY.StaticInjection) =
    get_output_offer_curves(PSY.get_operation_cost(device))
get_initial_input(::DecrementalOffer, device::PSY.StaticInjection) =
    PSY.get_decremental_initial_input(PSY.get_operation_cost(device))
get_initial_input(::IncrementalOffer, device::PSY.StaticInjection) =
    PSY.get_incremental_initial_input(PSY.get_operation_cost(device))

# direction and cost curve (needed for VOM code path):
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

_get_parameter_field(::StartupCostParameter, op_cost) = PSY.get_start_up(op_cost)
_get_parameter_field(::ShutdownCostParameter, op_cost) = PSY.get_shut_down(op_cost)
_get_parameter_field(::IncrementalCostAtMinParameter, op_cost) =
    PSY.get_incremental_initial_input(op_cost)
_get_parameter_field(::DecrementalCostAtMinParameter, op_cost) =
    PSY.get_decremental_initial_input(op_cost)
_get_parameter_field(
    ::Union{
        IncrementalPiecewiseLinearSlopeParameter,
        IncrementalPiecewiseLinearBreakpointParameter,
    },
    op_cost,
) = get_output_offer_curves(op_cost)
_get_parameter_field(
    ::Union{
        DecrementalPiecewiseLinearSlopeParameter,
        DecrementalPiecewiseLinearBreakpointParameter,
    },
    op_cost,
) = get_input_offer_curves(op_cost)

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
    label = string(dir)

    (initial_is_ts && !variable_is_ts) &&
        @warn "In `MarketBidCost` for $(get_name(device)), found time series for `$(label)_initial_input` but non-time-series `$(label)_offer_curves`; will ignore `initial_input` of `$(label)_offer_curves`"
    (variable_is_ts && !initial_is_ts) &&
        throw(
            ArgumentError(
                "In `MarketBidCost` for $(get_name(device)), if providing time series for `$(label)_offer_curves`, must also provide time series for `$(label)_initial_input`",
            ),
        )

    if !variable_is_ts && !initial_is_ts
        _validate_eltype(
            Union{Float64, Nothing}, device, initial_input, " $(label)_initial_input",
        )
    else
        _validate_eltype(
            Float64, device, initial_input, " $(label)_initial_input",
        )
    end
end

curvity_check(::IncrementalOffer, x) = PSY.is_convex(x)
curvity_check(::DecrementalOffer, x) = PSY.is_concave(x)
expected_curvity(::IncrementalOffer) = "convex"
expected_curvity(::DecrementalOffer) = "concave"

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
        _validate_eltype(expected_type, device, x, " $(string(dir)) offer curves")
        cost_curve_name = nameof(typeof(PSY.get_operation_cost(device)))
        curvity_check(dir, x) ||
            throw(
                ArgumentError(
                    "$(uppercasefirst(string(dir))) $cost_curve_name for component $(device_name) is non-$(expected_curvity(dir))",
                ),
            )

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
# Section 11: PWL Objective Terms + Variable Objective Formulation (generic)
# Load formulation overloads (AbstractControllablePowerLoadFormulation) are in POM.
#################################################################################

"""
Add PWL objective terms using the **delta (incremental/block-offer) formulation**.

Given an offer curve with breakpoints ``P_0, P_1, \\ldots, P_n`` and slopes
``m_1, m_2, \\ldots, m_n``, this function:

1. Creates delta variables ``\\delta_k \\geq 0`` for each segment via [`add_pwl_variables!`](@ref),
   with no upper bound (block sizes are enforced by constraints).
2. Adds linking and block-size constraints via [`_add_pwl_constraint!`](@ref):
   ``p = \\sum_k \\delta_k`` and ``\\delta_k \\leq P_{k+1} - P_k``.
3. Builds the objective expression ``C = \\sum_k m_k \\, \\delta_k`` via [`get_pwl_cost_expression`](@ref).

For convex offer curves (``m_1 \\leq m_2 \\leq \\cdots \\leq m_n``), no SOS2 or binary
variables are needed — the optimizer fills cheap segments first automatically.

Dispatches on `OfferDirection` (incremental or decremental) to select the appropriate
variable and constraint types.

See also: [`_add_pwl_term!`](@ref) for the lambda (convex combination) formulation used by
`CostCurve{PiecewisePointCurve}`.
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
    is_variant = is_time_variant(get_offer_curves(dir, component))
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

        if is_variant
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

#################################################################################
# Section 12: TimeSeriesValueCurve Objective Formulation
# PSY-free delta PWL objective for CostCurve{TimeSeriesPiecewiseIncrementalCurve}.
# Reads slopes/breakpoints from pre-populated parameter containers.
#################################################################################

"""
    add_variable_cost_to_objective!(container, ::T, component, cost_function, ::U; dir)

Objective function dispatch for `CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve}`.
Routes to the PSY-free delta PWL formulation that reads from parameter containers.
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::T,
    component::C,
    ::IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve},
    ::U;
    dir::OfferDirection = IncrementalOffer(),
) where {
    T <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    _add_ts_incremental_pwl_cost!(dir, container, component, T(), U())
    return
end

"""
PSY-free delta PWL objective formulation for time-series-backed incremental
value curves. Reads slopes/breakpoints from parameter containers populated
externally. All parameter array lookups and buffer allocations are hoisted
before the time loop to avoid repeated dictionary lookups and allocations.
"""
function _add_ts_incremental_pwl_cost!(
    dir::D,
    container::OptimizationContainer,
    component::C,
    ::T,
    ::U,
) where {
    D <: OfferDirection,
    C <: IS.InfrastructureSystemsComponent,
    T <: VariableType,
    U <: AbstractDeviceFormulation,
}
    W = _block_offer_var(dir)
    X = _block_offer_constraint(dir)
    name::String = get_name(component)
    dt::Float64 = Dates.value(get_resolution(container)) / MILLISECONDS_IN_HOUR
    sign_dt::Float64 = _objective_sign(dir) * dt
    model_base_power::Float64 = get_model_base_power(container)

    # Hoist parameter array lookups out of the time loop (4 dict lookups total, not 4*T)
    SlopeParam = _slope_param(dir)
    BPParam = _breakpoint_param(dir)
    slope_arr = get_parameter_array(container, SlopeParam(), C)
    slope_mult = get_parameter_multiplier_array(container, SlopeParam(), C)
    bp_arr = get_parameter_array(container, BPParam(), C)
    bp_mult = get_parameter_multiplier_array(container, BPParam(), C)

    # Pre-allocate buffers sized from the parameter array axes
    seg_axis = axes(slope_arr)[2]
    point_axis = axes(bp_arr)[2]
    n_segments = length(seg_axis)
    n_points = length(point_axis)
    @assert_op n_segments == n_points - 1
    slopes = Vector{Float64}(undef, n_segments)
    breakpoints = Vector{Float64}(undef, n_points)

    # NATURAL_UNITS conversion factors (slopes scale up, breakpoints scale down)
    inv_base::Float64 = 1.0 / model_base_power

    for t in get_time_steps(container)
        _fill_pwl_data_from_arrays!(
            slopes, breakpoints, slope_arr, slope_mult, bp_arr, bp_mult,
            seg_axis, point_axis, name, t, model_base_power, inv_base)
        pwl_vars::Vector{JuMP.VariableRef} = add_pwl_variables!(
            container, W, C, name, t, n_segments; upper_bound = Inf)
        _add_pwl_constraint!(container, component, T(), U(), breakpoints, pwl_vars, t, X)
        pwl_cost::JuMP.AffExpr = get_pwl_cost_expression(pwl_vars, slopes, sign_dt)
        add_cost_to_expression!(container, ProductionCostExpression, pwl_cost, C, name, t)
        add_to_objective_variant_expression!(container, pwl_cost)
    end
    return
end

"""
Fill pre-allocated slope and breakpoint buffers from parameter arrays for a single
time step. Applies NATURAL_UNITS conversion in-place (slopes × base_power,
breakpoints / base_power), avoiding intermediate DenseAxisArray allocations.
"""
function _fill_pwl_data_from_arrays!(
    slopes::Vector{Float64},
    breakpoints::Vector{Float64},
    slope_arr::DenseAxisArray{Float64},
    slope_mult::DenseAxisArray{Float64},
    bp_arr::DenseAxisArray{Float64},
    bp_mult::DenseAxisArray{Float64},
    seg_axis::Vector,
    point_axis::Vector,
    name::String,
    time::Int,
    model_base_power::Float64,
    inv_base::Float64,
)
    for (i, seg) in enumerate(seg_axis)
        slopes[i] = slope_arr[name, seg, time] * slope_mult[name, seg, time] *
                     model_base_power
    end
    for (i, pt) in enumerate(point_axis)
        breakpoints[i] = bp_arr[name, pt, time] * bp_mult[name, pt, time] * inv_base
    end
    return
end
