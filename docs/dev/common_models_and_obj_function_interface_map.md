# IOM `common_models/` and `objective_function/` : Interface Map

For these folders, the interfaces is fairly clean and sensible: architecturally sound and straightforward enough one doesn't get lost among what's-calling-what-from-where. 

This categorizes functions by their role in the IOM/POM split and only includes functions with nontrivial logic or important design implications.

---

## Category 1: Pure IOM Internal

These have real implementations and are never overridden or directly called by POM.

### Constraint math (`common_models/`)

- **`_add_bound_range_constraints_impl!`** (`range_constraint.jl:84`) — Inner loop for range constraints: iterates devices, calls `get_min_max_limits()`, creates JuMP constraints. Shared by `add_range_constraints!` for both variable and expression targets.
- **`_add_semicontinuous_bound_range_constraints_impl!`** (`range_constraint.jl:195,221`) — Same but multiplies bounds by `OnVariable` binary. Special-cases `ThermalGen` must-run (skips binary).
- **`_add_linear_ramp_constraints_impl!`** (`rateofchange_constraints.jl:129`) — Inner loop: creates ramp-up/down constraints with Big-M start/stop relaxation and optional slack variables.
- **`device_duration_retrospective!`**, `device_duration_look_ahead!`, `device_duration_parameters!`, `device_duration_compact_retrospective!` (`duration_constraints.jl`) — Four UC duration constraint formulations. All self-contained math, ~100 lines each.

### Cost curve internals (`objective_function/`)

- **`_add_pwl_term!`** (`piecewise_linear.jl`) — Full PWL cost pipeline per component: normalize, create delta vars, add linking/normalization/SOS2 constraints, build cost expression. ~80 lines.
- **`_add_quadraticcurve_variable_cost!`** (`quadratic_curve.jl`) — Loops timesteps, adds `var^2 * quad + var * prop` terms. Checks monotonicity.
- **`add_pwl_term!`** (market bid version, `market_bid.jl`) — Block-offer PWL: retrieves breakpoints/slopes (possibly time-variant from parameters), creates variables, adds constraints, routes to variant/invariant objective.
- **`process_market_bid_parameters!`** / **`process_import_export_parameters!`** (`market_bid.jl`) — Validate offer curve data, add startup/shutdown/slope/breakpoint parameter containers.

### Friction points (IOM-internal but device-aware)

- **`_onvar_cost`** (`common.jl:129-156`) — Extracts on-variable (fixed) cost from cost curves. Comment says "only called in POM, device specific code" — candidate to move.
- **`_sos_status`** / **`_include_min_gen_power_in_constraint`** (`piecewise_linear.jl`, `market_bid.jl`) — Reference `ThermalGen` and UC formulations directly inside IOM.

---

## Category 2: API (Implemented in IOM, Called by POM)

Workhorse functions POM calls from constructors but never overrides.

### Variable/constraint builders (`common_models/`)

| Function | File | What POM passes in |
|----------|------|--------------------|
| `add_variables!` | `add_variable.jl` | `(ActivePowerVariable, devices, formulation)` etc. |
| `add_range_constraints!` | `range_constraint.jl` | Constraint type + var/expression type |
| `add_semicontinuous_range_constraints!` | `range_constraint.jl` | Same, with `OnVariable` binary |
| `add_reserve_bound_range_constraints!` | `range_constraint.jl` | Reserve constraints with `ReservationVariable` |
| `add_parameterized_{lower,upper}_bound_range_constraints` | `range_constraint.jl` | Constraints with parameter-driven bounds |
| `add_linear_ramp_constraints!` | `rateofchange_constraints.jl` | Ramp constraint type + var type |
| `add_semicontinuous_ramp_constraints!` | `rateofchange_constraints.jl` | Ramp with start/stop relaxation |
| `add_param_container!` | `add_param_container.jl` | Parameter type dispatch (TS, Objective, VariableValue, Event, FixValue) |

### Cost builders (`objective_function/`)

| Function | File | What POM passes in |
|----------|------|--------------------|
| `add_variable_cost!` | `common.jl` | Var type + devices; dispatches on cost curve type internally |
| `add_proportional_cost!` | `proportional.jl` | For non-time-variant proportional costs |
| `add_proportional_cost_maybe_time_variant!` | `proportional.jl` | Checks `is_time_variant_term()` per timestep |
| `add_start_up_cost!` / `add_shut_down_cost!` | `start_up_shut_down.jl` | Startup/shutdown cost terms |
| `add_proportional_cost_invariant!` | `cost_term_helpers.jl` | Normalized linear cost across all timesteps |
| `add_cost_term_invariant!` / `add_cost_term_variant!` | `cost_term_helpers.jl` | Single-timestep cost term to invariant/variant objective |

The cost curve dispatch chain (`add_variable_cost!` -> `add_variable_cost_to_objective!` dispatched on `CostCurve{LinearCurve}`, `{QuadraticCurve}`, `{PiecewisePointCurve}`, `FuelCurve{*}`, `OfferCurveCost`) is entirely in IOM. POM never needs to know about it — it just calls `add_variable_cost!`.

---

## Category 3: Stubs / Extension Points (Defined in IOM, Implemented in POM)

All in `interfaces.jl` unless noted. The constraint/variable/objective related ones are quite stable and well-thought-out: i.e., actually needed. The other ones may be subject to change.

### Pure stubs (error by default)

| Function | POM implements for |
|----------|--------------------|
| `construct_device!` | Every device type x formulation x stage (~100+ methods) |
| `construct_service!` | Every service type x formulation |
| `build_container!` | `DecisionModel`, `EmulationModel` |
| `construct_services!` | Iterates services, calls `construct_service!` |
| `construct_network!` | CopperPlate, AreaBalance, PTDF, DC/AC power flow |
| `construct_hvdc_network!` | HVDC network formulations |
| `add_to_objective_function!` | Per device/service type |
| `add_constraints!` | Per constraint type x device type (~50+ methods) |
| `add_parameters!` | Per parameter type x device type |
| `get_expression_multiplier` | Per parameter x expression x device x formulation |
| `get_variable_binary` | Per variable type x device type |
| `proportional_cost` (both signatures) | Thermal, loads, reserves |
| `start_up_cost` | Thermal (handles StartUpStages, NamedTuple, Float64) |
| `make_system_expressions!` | Per network model type |
| `_get_initial_condition_type` | Ramp constraint IC types |

### Stubs with sensible defaults (POM overrides specific cases)

| Function | Default | POM overrides when |
|----------|---------|-------------------|
| `get_variable_multiplier` | `1.0` | Loads (negative), charge power, etc. |
| `get_variable_lower_bound` / `get_variable_upper_bound` | `nothing` | Hydro, storage, reserves, HVDC |
| `get_min_max_limits` | `nothing` | ~30+ methods across all device types |
| `get_default_attributes` | `Dict()` | Thermal (hot_start_slack etc.), hydro (many) |
| `get_default_time_series_names` | `Dict()` | Renewable, thermal, hydro TS parameter names |
| `is_time_variant_term` | `false` | MarketBidCost devices |
| `skip_proportional_cost` | `false` | `ThermalGen` with `must_run == true` |
| `objective_function_multiplier` | `1.0` | `-1.0` for loads |
| `variable_cost` | `get_variable(cost)` | Storage: maps `ActivePower{In/Out}` to charge/discharge |
| `get_multiplier_value(::TimeSeriesParameter, ...)` | `1.0` | Device-specific TS scaling |

---

## Category 4: Bulk in IOM, Thin POM Wrapper

POM's `add_to_objective_function!` methods are typically one-liners that call IOM:

```julia
# POM: thermal_generation.jl
function add_to_objective_function!(container, devices, model::DeviceModel{ThermalGen, ThermalStandardUC}, ::Type{<:AbstractPowerModel})
    add_variable_cost!(container, ActivePowerVariable(), devices, model)
    add_proportional_cost!(container, OnVariable(), devices, model)
    add_start_up_cost!(container, StartVariable(), devices, model)
    add_shut_down_cost!(container, StopVariable(), devices, model)
end
```

Similarly, POM's `construct_device!` methods are recipes that call IOM's API functions
(`add_variables!`, `add_range_constraints!`, `add_linear_ramp_constraints!`, etc.)
with specific type parameters. The actual math lives in IOM.

---

## Design Assessment

**Clean:** Cost curve dispatch, constraint math, variable creation, PWL machinery — all self-contained in IOM.

**Friction:** A few IOM functions reference `ThermalGen` or UC formulations directly (`_sos_status`, `_include_min_gen_power_in_constraint`, `_add_semicontinuous_bound_range_constraints_impl!`'s must-run special case, `_onvar_cost`). These are candidates to either move to POM or abstract behind another stub.
