# Quadratic Approximations

```@meta
CurrentModule = InfrastructureOptimizationModels
```

This module provides three methods for approximating ``x^2`` as a piecewise linear (PWL)
function within a MIP framework. Each method produces a JuMP `AffExpr` that can be used
in constraints or objectives wherever ``x^2`` would appear.

All three methods share a common interface: they accept a vector of component names, a
range of time steps, a container of ``x`` variables, domain bounds, an approximation
resolution parameter, and a `meta` string that identifies the variable type being
approximated. The `meta` parameter allows creating multiple independent approximations
for the same component type (e.g., one for active power and another for reactive power).

## Solver SOS2

Approximates ``x^2`` using a convex combination of breakpoints with solver-native
[SOS2 constraints](https://en.wikipedia.org/wiki/Special_ordered_set) for adjacency
enforcement.

### Formulation

Given ``N`` uniformly spaced breakpoints ``x_1, \ldots, x_N`` on ``[x_{\min}, x_{\max}]``:

**Variables:**

```math
\lambda_i \in [0, 1], \quad i = 1, \ldots, N
```

**Constraints:**

```math
\begin{aligned}
x &= \sum_{i=1}^{N} \lambda_i \, x_i && \text{(linking)} \\
\sum_{i=1}^{N} \lambda_i &= 1 && \text{(normalization)} \\
\lambda &\in \text{SOS2} && \text{(adjacency, solver-native)}
\end{aligned}
```

**Approximation:**

```math
\hat{x}^2 = \sum_{i=1}^{N} \lambda_i \, x_i^2
```

### Complexity

For ``S`` segments (``N = S + 1`` breakpoints): ``N`` continuous variables, ``N + 1``
constraints, plus one SOS2 set per component per time step. The approximation error
decreases as ``O(1/S^2)``.

### Variables and Constraints

| Container Type | Description |
|:---|:---|
| [`QuadraticApproxVariable`](@ref) | Lambda (``\lambda``) convex combination weights |
| [`QuadraticApproxLinkingConstraint`](@ref) | Links ``x`` to weighted breakpoints |
| [`QuadraticApproxNormalizationConstraint`](@ref) | Ensures ``\sum \lambda_i = 1`` |

## Manual SOS2

Identical approximation to Solver SOS2, but replaces the solver-native SOS2 constraint
with explicit binary variables and linear inequality constraints. This is useful for
solvers that do not support SOS2 constraints natively.

### Additional Formulation

In addition to the linking and normalization constraints above, manual SOS2 introduces
``N - 1`` binary segment-selection variables ``z_j \in \{0, 1\}`` and enforces adjacency
via:

```math
\begin{aligned}
\sum_{j=1}^{N-1} z_j &= 1 && \text{(exactly one active segment)} \\
\lambda_1 &\leq z_1 \\
\lambda_i &\leq z_{i-1} + z_i && i = 2, \ldots, N-1 \\
\lambda_N &\leq z_{N-1}
\end{aligned}
```

### Complexity

For ``S`` segments: ``N`` continuous variables + ``S`` binary variables, ``N + S + 2``
linear constraints per component per time step. Same approximation quality as Solver SOS2.

### Variables and Constraints

| Container Type | Description |
|:---|:---|
| [`QuadraticApproxVariable`](@ref) | Lambda (``\lambda``) convex combination weights |
| [`ManualSOS2BinaryVariable`](@ref) | Binary segment-selection variables (``z``) |
| [`QuadraticApproxLinkingConstraint`](@ref) | Links ``x`` to weighted breakpoints |
| [`QuadraticApproxNormalizationConstraint`](@ref) | Ensures ``\sum \lambda_i = 1`` |
| [`ManualSOS2SegmentSelectionConstraint`](@ref) | Ensures ``\sum z_j = 1`` |

## Sawtooth

Approximates ``x^2`` using the recursive sawtooth MIP formulation from
Beach, Burlacu, Hager, and Hildebrand (2024). This method requires only ``O(\log(1/\varepsilon))``
binary variables to achieve error ``\varepsilon``, compared to ``O(1/\sqrt{\varepsilon})``
for the SOS2 methods.

### Formulation

Given depth ``L`` and ``\Delta = x_{\max} - x_{\min}``:

**Variables:**

```math
\begin{aligned}
g_j &\in [0, 1], \quad j = 0, \ldots, L && \text{(auxiliary continuous)} \\
\alpha_j &\in \{0, 1\}, \quad j = 1, \ldots, L && \text{(binary)}
\end{aligned}
```

**Constraints:**

```math
\begin{aligned}
g_0 &= \frac{x - x_{\min}}{\Delta} && \text{(linking)} \\
g_j &\leq 2\, g_{j-1} && j = 1, \ldots, L \\
g_j &\leq 2\,(1 - g_{j-1}) && j = 1, \ldots, L \\
g_j &\geq 2\,(g_{j-1} - \alpha_j) && j = 1, \ldots, L \\
g_j &\geq 2\,(\alpha_j - g_{j-1}) && j = 1, \ldots, L
\end{aligned}
```

**Approximation:**

```math
\hat{x}^2 = x_{\min}^2 + (2\, x_{\min}\, \Delta + \Delta^2)\, g_0 - \sum_{j=1}^{L} \Delta^2 \cdot 2^{-2j}\, g_j
```

### Complexity

For depth ``L``: ``L + 1`` continuous variables + ``L`` binary variables, ``4L + 1``
constraints per component per time step. The approximation interpolates ``x^2`` at
``2^L + 1`` uniformly spaced breakpoints with maximum overestimation error
``\Delta^2 \cdot 2^{-2L-2}``.

### Variables and Constraints

| Container Type | Description |
|:---|:---|
| [`SawtoothAuxVariable`](@ref) | Auxiliary continuous variables (``g_0, \ldots, g_L``) |
| [`SawtoothBinaryVariable`](@ref) | Binary variables (``\alpha_1, \ldots, \alpha_L``) |
| [`SawtoothLinkingConstraint`](@ref) | Links ``g_0`` to normalized ``x`` |

## Comparison

| Property | Solver SOS2 | Manual SOS2 | Sawtooth |
|:---|:---|:---|:---|
| Binary variables | 0 | ``S`` | ``L`` |
| Continuous variables | ``S + 1`` | ``S + 1`` | ``L + 1`` |
| Breakpoints | ``S + 1`` | ``S + 1`` | ``2^L + 1`` |
| Max error | ``O(\Delta^2 / S^2)`` | ``O(\Delta^2 / S^2)`` | ``\Delta^2 \cdot 2^{-2L-2}`` |
| Solver requirements | SOS2 support | MIP only | MIP only |

To match the number of breakpoints between methods, set ``S = 2^L``. At equal breakpoint
count the approximation quality is identical, but the sawtooth uses ``L = \log_2 S``
binary variables instead of ``S``.
