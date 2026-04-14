# Reduce Compilation Times and Runtime Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate type instability and dynamic dispatch from hot paths in model construction, solve loops, and parameter updates to reduce compilation times and runtime overhead.

**Architecture:** Changes are organized in three phases by impact. Phase 1 addresses abstract container field types in `OptimizationContainer` and `ObjectiveFunction` — the highest-ROI changes. Phase 2 eliminates per-solve-step allocations and type instability. Phase 3 improves model-build-time patterns. Tasks within each phase that touch different files are independent and can be dispatched to parallel agents.

**Tech Stack:** Julia 1.10+, JuMP, MathOptInterface, OrderedCollections

**Testing:** Run `julia --project=test test/runtests.jl` (1201 unit tests + 10 Aqua.jl checks). Run formatter after each task: `julia scripts/formatter/formatter_code.jl`.

**Sienna conventions:** See `.claude/Sienna.md` — never use `isa`/`<:` for branching, always use `--project=test`, run formatter after every change.

---

## Parallelism Map

Tasks touching different files can run as parallel agents. Tasks touching the same file must be sequential.

```
Phase 1 (sequential — all touch optimization_container.jl):
  Task 1 → Task 2 → Task 3 → Task 4

Phase 2 (parallel — independent files):
  Task 5  (parameter_container.jl)        ─┐
  Task 6  (store_common.jl)               ─┤
  Task 7  (jump_utils.jl)                 ─┤── all parallel
  Task 8  (settings.jl)                   ─┘

Phase 3 (parallel — independent files):
  Task 9  (duration_constraints.jl)       ─┐
  Task 10 (rateofchange_constraints.jl)   ─┤── all parallel
  Task 11 (abstract_model_store.jl)       ─┘
```

---

## Phase 1: Container Type Narrowing (Sequential — same file)

### Task 1: Narrow `AbstractArray` dict values to concrete union

**Files:**
- Modify: `src/core/optimization_container.jl:1-10` (PrimalValuesCache)
- Modify: `src/core/optimization_container.jl:63-120` (OptimizationContainer struct + constructor)
- Modify: `src/core/initial_conditions.jl:62-75` (InitialConditionsData)
- Test: `test/test_optimization_container.jl`

**Context:** The five container dicts (`variables`, `aux_variables`, `duals`, `constraints`, `expressions`) and `PrimalValuesCache` and `InitialConditionsData` all use `AbstractArray` as the value type. Every access returns `AbstractArray`, forcing runtime dispatch on all downstream operations (`jump_value.()`, `getindex`, broadcasting). The actual concrete types stored are always `JuMP.Containers.DenseAxisArray` or `JuMP.Containers.SparseAxisArray`.

- [ ] **Step 1: Define the concrete union type alias**

Add near the top of `optimization_container.jl`, after imports but before struct definitions:

```julia
const JuMPArray = Union{JuMP.Containers.DenseAxisArray, JuMP.Containers.SparseAxisArray}
```

- [ ] **Step 2: Update `PrimalValuesCache`**

Change `src/core/optimization_container.jl` lines 1-9:

```julia
struct PrimalValuesCache
    variables_cache::Dict{VariableKey, JuMPArray}
    expressions_cache::Dict{ExpressionKey, JuMPArray}
end

function PrimalValuesCache()
    return PrimalValuesCache(
        Dict{VariableKey, JuMPArray}(),
        Dict{ExpressionKey, JuMPArray}(),
    )
end
```

- [ ] **Step 3: Update `OptimizationContainer` struct fields**

Change the five `AbstractArray` fields in the struct definition (lines ~67-72):

```julia
    variables::OrderedDict{VariableKey, JuMPArray}
    aux_variables::OrderedDict{AuxVarKey, JuMPArray}
    duals::OrderedDict{ConstraintKey, JuMPArray}
    constraints::OrderedDict{ConstraintKey, JuMPArray}
    ...
    expressions::OrderedDict{ExpressionKey, JuMPArray}
```

And update the constructor to match:

```julia
    OrderedDict{VariableKey, JuMPArray}(),
    OrderedDict{AuxVarKey, JuMPArray}(),
    OrderedDict{ConstraintKey, JuMPArray}(),
    OrderedDict{ConstraintKey, JuMPArray}(),
    ...
    OrderedDict{ExpressionKey, JuMPArray}(),
```

Also update `infeasibility_conflict::Dict{Symbol, Array}` to `Dict{Symbol, JuMPArray}` if it stores JuMP arrays, or leave as-is if it stores plain `Array`.

- [ ] **Step 4: Update `InitialConditionsData`**

Change `src/core/initial_conditions.jl` lines 62-75:

```julia
mutable struct InitialConditionsData
    duals::Dict{ConstraintKey, JuMPArray}
    parameters::Dict{ParameterKey, JuMPArray}
    variables::Dict{VariableKey, JuMPArray}
    aux_variables::Dict{AuxVarKey, JuMPArray}
end

function InitialConditionsData()
    return InitialConditionsData(
        Dict{ConstraintKey, JuMPArray}(),
        Dict{ParameterKey, JuMPArray}(),
        Dict{VariableKey, JuMPArray}(),
        Dict{AuxVarKey, JuMPArray}(),
    )
end
```

- [ ] **Step 5: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All 1201 tests pass. If any `convert` errors arise, it means some code path stores a non-DenseAxisArray/SparseAxisArray value — investigate and widen the union if needed.

- [ ] **Step 6: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/core/optimization_container.jl src/core/initial_conditions.jl
git commit -m "perf: narrow AbstractArray dict values to concrete DenseAxisArray|SparseAxisArray union"
```

---

### Task 2: Type `ObjectiveFunction.invariant_terms` as concrete union

**Files:**
- Modify: `src/core/optimization_container.jl:17-61` (ObjectiveFunction struct)
- Modify: `src/core/optimization_container.jl:1219-1232` (add_to_objective_invariant_expression!)
- Test: `test/test_optimization_container.jl`

**Context:** `invariant_terms::JuMP.AbstractJuMPScalar` forces dynamic dispatch on every `add_to_objective_invariant_expression!` call. The field only ever holds `GenericAffExpr{Float64, VariableRef}` or `GenericQuadExpr{Float64, VariableRef}`. A concrete 2-member union enables compile-time union splitting. The existing runtime `typeof` check at line 1223 confirms the instability.

- [ ] **Step 1: Define the concrete union alias**

Add near the `JuMPArray` alias:

```julia
const JuMPScalarExpr = Union{
    JuMP.GenericAffExpr{Float64, JuMP.VariableRef},
    JuMP.GenericQuadExpr{Float64, JuMP.VariableRef},
}
```

- [ ] **Step 2: Update `ObjectiveFunction` struct**

Change the field type and inner constructor:

```julia
mutable struct ObjectiveFunction
    invariant_terms::JuMPScalarExpr
    variant_terms::GAE
    synchronized::Bool
    sense::MOI.OptimizationSense
    function ObjectiveFunction(invariant_terms::JuMPScalarExpr,
        variant_terms::GAE,
        synchronized::Bool,
        sense::MOI.OptimizationSense = MOI.MIN_SENSE)
        new(invariant_terms, variant_terms, synchronized, sense)
    end
end
```

- [ ] **Step 3: Simplify `add_to_objective_invariant_expression!`**

The runtime `typeof` check is no longer needed — Julia will union-split automatically. However, the promotion from `AffExpr` to `QuadExpr` when adding a quadratic term requires updating the field. Keep the existing logic but remove the `typeof` indirection:

```julia
function add_to_objective_invariant_expression!(
    container::OptimizationContainer,
    cost_expr::T,
) where {T <: JuMP.AbstractJuMPScalar}
    obj = container.objective_function
    if obj.invariant_terms isa JuMP.GenericAffExpr && T <: JuMP.GenericQuadExpr
        container.objective_function.invariant_terms += cost_expr
    else
        JuMP.add_to_expression!(obj.invariant_terms, cost_expr)
    end
    return
end
```

Note: The `isa` check here is necessary because it guards a type-promotion mutation of the field. This is one of the rare cases where `isa` is acceptable per Sienna conventions — it's not branching on business logic, it's handling JuMP's type promotion.

- [ ] **Step 4: Update `get_objective_expression`**

In the same struct's accessor (lines ~36-47), the `isa` check on `invariant_terms` is already present. With the concrete union, Julia will union-split it automatically. Leave the logic as-is — it will now compile efficiently.

- [ ] **Step 5: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 6: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/core/optimization_container.jl
git commit -m "perf: type ObjectiveFunction.invariant_terms as concrete AffExpr|QuadExpr union"
```

---

### Task 3: Type PWL variables container with `JuMP.VariableRef`

**Files:**
- Modify: `src/core/optimization_container.jl:679-682` (_get_pwl_variables_container)
- Test: `test/test_optimization_container.jl`

**Context:** `_get_pwl_variables_container()` creates a `SparseAxisArray` backed by `Dict{Tuple{String,Int,Int}, Any}`. Every access in the per-timestep objective loop returns `Any`, defeating specialization of `JuMP.add_to_expression!`.

- [ ] **Step 1: Change the Dict value type to `JuMP.VariableRef`**

```julia
function _get_pwl_variables_container()
    contents = Dict{Tuple{String, Int, Int}, JuMP.VariableRef}()
    return SparseAxisArray(contents)
end
```

- [ ] **Step 2: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass. If any test fails because a non-VariableRef is stored, investigate the caller.

- [ ] **Step 3: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/core/optimization_container.jl
git commit -m "perf: type PWL variables container values as JuMP.VariableRef"
```

---

### Task 4: Cache aux variable key partitions at build time

**Files:**
- Modify: `src/core/optimization_container.jl:63-120` (add cached fields to struct)
- Modify: `src/core/optimization_container.jl:1256-1273` (calculate_aux_variables!)
- Test: `test/test_optimization_container.jl`

**Context:** `calculate_aux_variables!` runs every solve step and re-computes `filter(is_from_power_flow ∘ get_entry_type, keys(...))` and `setdiff(...)` on invariant key sets. These allocations are unnecessary — the key sets never change after model build.

- [ ] **Step 1: Add cached key vector fields to `OptimizationContainer`**

Add two new fields to the struct after `built_for_recurrent_solves`:

```julia
    pf_aux_var_keys::Vector{AuxVarKey}
    non_pf_aux_var_keys::Vector{AuxVarKey}
```

Initialize them as empty vectors in the constructor:

```julia
    AuxVarKey[],
    AuxVarKey[],
```

- [ ] **Step 2: Add a function to populate the cached keys**

Add a function that should be called at the end of model build (after all aux variables are added):

```julia
function _cache_aux_variable_key_partitions!(container::OptimizationContainer)
    aux_var_keys = keys(get_aux_variables(container))
    pf_keys = filter(is_from_power_flow ∘ get_entry_type, aux_var_keys)
    container.pf_aux_var_keys = collect(pf_keys)
    container.non_pf_aux_var_keys = collect(setdiff(aux_var_keys, pf_keys))
    return
end
```

Find where the model build finalizes aux variables (search for `built_for_recurrent_solves = true` or the end of the build sequence) and call `_cache_aux_variable_key_partitions!(container)` there.

- [ ] **Step 3: Update `calculate_aux_variables!` to use cached keys**

Replace the allocation-heavy lines:

```julia
function calculate_aux_variables!(
    container::OptimizationContainer,
    system::IS.InfrastructureSystemsContainer,
)
    pf_aux_var_keys = container.pf_aux_var_keys
    non_pf_aux_var_keys = container.non_pf_aux_var_keys
    @assert isempty(pf_aux_var_keys) || !isempty(get_power_flow_evaluation_data(container))
    # ... rest unchanged
```

- [ ] **Step 4: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass. If cached keys are empty (cache function not called), some tests may fail — verify the cache function is called at the right point in the build sequence.

- [ ] **Step 5: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/core/optimization_container.jl
git commit -m "perf: cache aux variable key partitions at build time to avoid per-step allocations"
```

---

## Phase 2: Per-Solve-Step Overhead (Parallel — independent files)

### Task 5: Parameterize `ParameterContainer` on attribute type

**Files:**
- Modify: `src/core/parameter_container.jl:97-110` (struct definition)
- Modify: `src/core/optimization_container.jl:73` (parameters dict value type — only if needed)
- Test: `test/test_optimization_container.jl`

**Context:** `attributes::ParameterAttributes` is abstract. `calculate_parameter_values` and `get_parameter_values` dispatch dynamically on each access during rolling-horizon solve steps. Adding a type parameter `A` lets the compiler specialize.

- [ ] **Step 1: Add type parameter `A` to `ParameterContainer`**

```julia
struct ParameterContainer{T <: AbstractArray, U <: AbstractArray, A <: ParameterAttributes}
    attributes::A
    parameter_array::T
    multiplier_array::U
end

function ParameterContainer(parameter_array, multiplier_array)
    return ParameterContainer(NoAttributes(), parameter_array, multiplier_array)
end
```

- [ ] **Step 2: Check callers that construct `ParameterContainer` explicitly**

Search for all `ParameterContainer(` calls in `src/`. The third type parameter `A` should be inferred automatically from the constructor argument. Verify no call site explicitly specifies `ParameterContainer{T, U}(...)` — if any do, update them to `ParameterContainer{T, U, typeof(attributes)}(...)` or just let Julia infer.

- [ ] **Step 3: Check the dict in OptimizationContainer**

The field `parameters::OrderedDict{ParameterKey, ParameterContainer}` uses the unparameterized `ParameterContainer`. Since different parameter containers will have different `A` types, the dict must store a supertype. `ParameterContainer` without parameters is already abstract in Julia's type system for a 3-parameter struct — this is fine. The dict value type should remain `ParameterContainer` (which Julia treats as `ParameterContainer{T,U,A} where {T,U,A}`). The key benefit is that when a specific container is retrieved and used, the compiler knows the concrete `A` type from that point forward.

- [ ] **Step 4: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 5: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/core/parameter_container.jl
git commit -m "perf: parameterize ParameterContainer on attribute type for dispatch specialization"
```

---

### Task 6: Replace `export_params` dict with typed struct

**Files:**
- Modify: `src/operation/store_common.jl:27-34` (construction site)
- Modify: `src/operation/store_common.jl` (all access sites in `write_model_*_outputs!` functions)
- Test: `test/test_optimization_container.jl`

**Context:** `export_params::Dict{Symbol, Any}` forces every field lookup to return `Any` in 5 `write_model_*_outputs!` functions called per solve step. A typed struct eliminates both the dict hashing and the type instability.

- [ ] **Step 1: Define `ExportParameters` struct**

Add at the top of `store_common.jl` (or in a nearby definitions file):

```julia
struct ExportParameters
    exports::Exports
    exports_path::String
    file_type::Type
    resolution::Dates.Millisecond
    horizon_count::Int
end
```

Check the actual types used for `exports`, `file_type` by reading the construction site. Adjust the field types to match what `get_export_file_type` and `get_resolution` return.

- [ ] **Step 2: Replace the Dict construction**

Change lines 27-34:

```julia
export_params = ExportParameters(
    exports,
    joinpath(exports.path, _sanitize_model_name(string(get_name(model)))),
    get_export_file_type(exports),
    get_resolution(model),
    get_horizon(get_settings(model)) ÷ get_resolution(model),
)
```

- [ ] **Step 3: Update all access sites**

Replace `export_params[:exports]` with `export_params.exports`, `export_params[:exports_path]` with `export_params.exports_path`, etc. Search for all `export_params[` occurrences in `store_common.jl` and replace with field access.

- [ ] **Step 4: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 5: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/operation/store_common.jl
git commit -m "perf: replace Dict{Symbol,Any} export_params with typed ExportParameters struct"
```

---

### Task 7: Factor `try-catch` out of `_get_solver_time` hot path

**Files:**
- Modify: `src/utils/jump_utils.jl:568-588`
- Test: `test/test_jump_utils.jl`

**Context:** `_get_solver_time` contains a `try-catch` that prevents inlining of the entire function, even though the `try` only fires once (first call). Factoring the probe into a separate function lets the compiler inline the fast path.

- [ ] **Step 1: Split into two functions**

Replace the existing `_get_solver_time` with:

```julia
function _probe_solver_time!(jump_model::JuMP.Model)
    try
        solver_solve_time = MOI.get(jump_model, MOI.SolveTimeSec())
        jump_model.ext[:try_supports_solvetime] = (trycatch = false, supports = true)
        return solver_solve_time
    catch
        @debug "SolveTimeSec() property not supported by the Solver"
        jump_model.ext[:try_supports_solvetime] = (trycatch = false, supports = false)
        return NaN
    end
end

function _get_solver_time(jump_model::JuMP.Model)
    try_s =
        get!(jump_model.ext, :try_supports_solvetime, (trycatch = true, supports = true))
    if try_s.trycatch
        return _probe_solver_time!(jump_model)
    elseif try_s.supports
        return MOI.get(jump_model, MOI.SolveTimeSec())
    else
        return NaN
    end
end
```

- [ ] **Step 2: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 3: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/utils/jump_utils.jl
git commit -m "perf: factor try-catch out of _get_solver_time fast path for inlining"
```

---

### Task 8: Narrow `Settings.optimizer` type

**Files:**
- Modify: `src/core/settings.jl:7` (field type)
- Test: `test/test_settings.jl`

**Context:** `optimizer::Any` causes runtime dispatch at model finalization. The constructor already coerces `DataType` inputs to `MOI.OptimizerWithAttributes`, so the stored value is always `nothing` or `MOI.OptimizerWithAttributes`.

- [ ] **Step 1: Change the field type**

```julia
    optimizer::Union{Nothing, MOI.OptimizerWithAttributes}
```

- [ ] **Step 2: Verify the constructor handles the coercion**

The constructor at lines ~55-58 already does:
```julia
if isa(optimizer, DataType)
    optimizer_ = MOI.OptimizerWithAttributes(optimizer)
else
    optimizer_ = optimizer
end
```

This means by the time the struct is constructed, the value is `nothing` or `MOI.OptimizerWithAttributes`. The type annotation will enforce this. Check if any other code path sets `optimizer` to something else — search for `settings.optimizer =` or `Settings(` calls.

- [ ] **Step 3: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass. If any test passes a non-`DataType`, non-`OptimizerWithAttributes`, non-`nothing` optimizer, it will fail with a `MethodError` — investigate and fix.

- [ ] **Step 4: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/core/settings.jl
git commit -m "perf: narrow Settings.optimizer from Any to Union{Nothing, OptimizerWithAttributes}"
```

---

## Phase 3: Model-Build Patterns (Parallel — independent files)

### Task 9: Clamp duration constraint range bounds

**Files:**
- Modify: `src/common_models/duration_constraints.jl` (6 sites across 4 functions)

**Context:** Inner loops construct a `UnitRange` then check `if i in time_steps` on each element. Since `time_steps` is always `1:T`, this is equivalent to clamping the lower bound to `max(1, t - d + 1)`. Eliminating the conditional removes a branch from the innermost loop.

- [ ] **Step 1: Identify all 6 sites**

Search for `if i in time_steps` in `duration_constraints.jl`. Each occurrence follows the pattern:

```julia
for i in UnitRange{Int}(Int(t - duration_data[ix].up + 1), t)
    if i in time_steps
        JuMP.add_to_expression!(lhs_on, varstart[name, i])
    end
end
```

- [ ] **Step 2: Replace each with clamped range**

Replace each occurrence with:

```julia
for i in max(first(time_steps), t - duration_data[ix].up + 1):t
    JuMP.add_to_expression!(lhs_on, varstart[name, i])
end
```

Adjust `.up` vs `.down` and `varstart` vs `varstop` per site. The key transformation is:
- `UnitRange{Int}(Int(t - d + 1), t)` + `if i in time_steps` becomes `max(first(time_steps), t - d + 1):t`
- No `Int()` conversion needed — arithmetic on `Int` already produces `Int`
- No `UnitRange{Int}()` constructor needed — `a:b` produces `UnitRange{Int}` when `a` and `b` are `Int`

- [ ] **Step 3: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 4: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/common_models/duration_constraints.jl
git commit -m "perf: clamp duration constraint range bounds instead of filtering by membership"
```

---

### Task 10: Pre-build IC name dict for ramp constraints

**Files:**
- Modify: `src/common_models/rateofchange_constraints.jl:215-243`

**Context:** `findfirst(ic -> get_component_name(ic) == name, initial_conditions_power)` creates a closure that captures the reassigned `name` variable (causing boxing) and performs O(N) linear search per device, making the outer loop O(N^2). A pre-built dict eliminates both problems.

- [ ] **Step 1: Find all `findfirst` + closure patterns in the file**

Search for `findfirst(ic ->` in `rateofchange_constraints.jl`. There may be multiple occurrences across different functions.

- [ ] **Step 2: Replace with dict lookup**

Before the `for dev in ramp_devices` loop, add:

```julia
ic_power_by_name = Dict(
    get_component_name(ic) => get_value(ic) for ic in initial_conditions_power
)
```

Then replace:
```julia
ic_idx = findfirst(ic -> get_component_name(ic) == name, initial_conditions_power)
ic_power = get_value(initial_conditions_power[ic_idx])
```

With:
```julia
ic_power = ic_power_by_name[name]
```

- [ ] **Step 3: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 4: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/common_models/rateofchange_constraints.jl
git commit -m "perf: replace O(N^2) findfirst closure with pre-built name dict in ramp constraints"
```

---

### Task 11: Fix `get_data_field` runtime symbol dispatch

**Files:**
- Modify: `src/core/abstract_model_store.jl:55-60`

**Context:** `get_data_field(store, type::Symbol) = getproperty(store, type)` uses a runtime symbol, preventing constant-folding. There's already a FIXME comment acknowledging this. The fix is to use `Val`-based dispatch so the symbol is a type parameter.

- [ ] **Step 1: Add `Val`-based dispatch**

Replace the existing function:

```julia
# Original with runtime symbol (kept as fallback):
# get_data_field(store::AbstractModelStore, type::Symbol) = getproperty(store, type)

# Val-based dispatch for compile-time resolution:
get_data_field(store::AbstractModelStore, ::Val{S}) where {S} = getfield(store, S)

# Convenience wrapper that converts Symbol to Val:
get_data_field(store::AbstractModelStore, type::Symbol) = get_data_field(store, Val(type))
```

- [ ] **Step 2: Update callers to use Val where the symbol is known at compile time**

Search for `get_data_field(` in `src/`. If any caller passes a literal symbol like `get_data_field(store, :variables)`, change it to `get_data_field(store, Val(:variables))` for compile-time resolution. If the symbol is computed at runtime, leave as-is — the wrapper will handle it.

- [ ] **Step 3: Run tests**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 4: Run formatter and commit**

```bash
julia scripts/formatter/formatter_code.jl
git add src/core/abstract_model_store.jl
git commit -m "perf: add Val-based dispatch for get_data_field to enable compile-time field resolution"
```

---

## Verification

After all tasks are complete:

- [ ] **Full test suite**: `julia --project=test test/runtests.jl` — all 1201+ tests pass
- [ ] **Formatter**: `julia scripts/formatter/formatter_code.jl` — no outstanding changes
- [ ] **Spot-check type stability**: Run `@code_warntype` on key functions to verify improvements:
  ```julia
  # In a Julia session with --project=test:
  using InfrastructureOptimizationModels
  # Check that container access no longer returns AbstractArray
  # Check that ObjectiveFunction methods are union-split
  # Check that _get_solver_time fast path is inlined
  ```
