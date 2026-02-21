# InfrastructureOptimizationModels.jl

Library for Optimization modeling in Sienna. This is a utility library that defines useful objects and
routines for managing power optimization models. Julia compat: `^1.10`.

> **General Sienna Programming Practices:** For information on performance requirements, code conventions, documentation practices, and contribution workflows that apply across all Sienna packages, see [Sienna.md](Sienna.md). Always check this file before making any changes or proposing to run any code.

## Type and Function Conventions

**Prefer IS types over PSY types:** When possible, use InfrastructureSystems parent types:
- `PSY.Component` → `IS.InfrastructureSystemsComponent`
- `PSY.System` → `IS.InfrastructureSystemsContainer`
- Cost curves: `IS.CostCurve`, `IS.LinearCurve`, `IS.UnitSystem`, etc.

## Optimization Container Keys

**Never create container keys directly.** Constraint keys, variable keys, aux_variable keys, parameter
keys, and any other optimization container keys must only be created through the corresponding
`add_constraints_container!`, `add_variables_container!`, `add_aux_variables_container!`, or
`add_parameters_container!` functions. Do not instantiate keys outside of these `add_foo_container`
functions.

## Testing

**Test file structure:** Test files are included by `InfrastructureOptimizationModelsTests.jl`, which
handles imports and mock infrastructure. Don't add `using`, `include`, or `const` alias statements
at the top of individual test files.

**Use mocks over PSY types:** Tests should use mock components (`MockThermalGen`, `MockSystem`, etc.)
rather than PowerSystems types when possible.
