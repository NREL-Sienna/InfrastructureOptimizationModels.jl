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

| Container Type                    | Description                                     |
|:--------------------------------- |:----------------------------------------------- |
| [`QuadraticVariable`](@ref) | Lambda (``\lambda``) convex combination weights |
| [`SOS2LinkingConstraint`](@ref)   | Links ``x`` to weighted breakpoints             |
| [`SOS2NormConstraint`](@ref)      | Ensures ``\sum \lambda_i = 1``                  |

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

| Container Type                                 | Description                                     |
|:---------------------------------------------- |:----------------------------------------------- |
| [`QuadraticVariable`](@ref)              | Lambda (``\lambda``) convex combination weights |
| [`ManualSOS2BinaryVariable`](@ref)             | Binary segment-selection variables (``z``)      |
| [`SOS2LinkingConstraint`](@ref)                | Links ``x`` to weighted breakpoints             |
| [`SOS2NormConstraint`](@ref)                   | Ensures ``\sum \lambda_i = 1``                  |
| [`ManualSOS2SegmentSelectionConstraint`](@ref) | Ensures ``\sum z_j = 1``                        |

## Sawtooth

Approximates ``x^2`` using the recursive sawtooth MIP formulation from
Beach, Burlacu, Hager, and Hildebrand (2023). This method requires only ``O(\log(1/\varepsilon))``
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

| Container Type                      | Description                                           |
|:----------------------------------- |:----------------------------------------------------- |
| [`SawtoothAuxVariable`](@ref)       | Auxiliary continuous variables (``g_0, \ldots, g_L``) |
| [`SawtoothBinaryVariable`](@ref)    | Binary variables (``\alpha_1, \ldots, \alpha_L``)     |
| [`SawtoothLinkingConstraint`](@ref) | Links ``g_0`` to normalized ``x``                     |

## Comparison

| Property             | Solver SOS2           | Manual SOS2           | Sawtooth                     |
|:-------------------- |:--------------------- |:--------------------- |:---------------------------- |
| Binary variables     | 0                     | ``S``                 | ``L``                        |
| Continuous variables | ``S + 1``             | ``S + 1``             | ``L + 1``                    |
| Breakpoints          | ``S + 1``             | ``S + 1``             | ``2^L + 1``                  |
| Max error            | ``O(\Delta^2 / S^2)`` | ``O(\Delta^2 / S^2)`` | ``\Delta^2 \cdot 2^{-2L-2}`` |
| Solver requirements  | SOS2 support          | MIP only              | MIP only                     |

To match the number of breakpoints between methods, set ``S = 2^L``. At equal breakpoint
count the approximation quality is identical, but the sawtooth uses ``L = \log_2 S``
binary variables instead of ``S``.

## Error Scaling

Both SOS2 and sawtooth produce the same PWL interpolation of ``x^2`` when they use the
same number of uniform breakpoints. With ``n`` uniform segments on ``[0, 1]``, the
interpolant ``F_n(x)`` satisfies the classical error bound (Barmann et al., 2023):

```math
0 \leq F_n(x) - x^2 \leq \frac{1}{4n^2} \quad \text{for all } x \in [0, 1]
```

The maximum error is attained at the midpoint of each segment. Since both methods
interpolate at the same breakpoints, the pointwise error is identical — the difference
lies entirely in how efficiently the segments are encoded.

### Same number of binary variables

If we budget ``L`` binary/integer variables for each method, SOS2 gets ``L`` segments
while sawtooth gets ``2^L`` segments. The error gap grows exponentially:

| ``L`` | SOS2 segments | SOS2 error            | Sawtooth segments | Sawtooth error        | Ratio |
|:----- |:------------- |:--------------------- |:----------------- |:--------------------- |:----- |
| 1     | 1             | ``2.50\times10^{-1}`` | 2                 | ``6.25\times10^{-2}`` | 4x    |
| 2     | 2             | ``6.25\times10^{-2}`` | 4                 | ``1.56\times10^{-2}`` | 4x    |
| 3     | 3             | ``2.78\times10^{-2}`` | 8                 | ``3.91\times10^{-3}`` | 7x    |
| 4     | 4             | ``1.56\times10^{-2}`` | 16                | ``9.77\times10^{-4}`` | 16x   |
| 5     | 5             | ``1.00\times10^{-2}`` | 32                | ``2.44\times10^{-4}`` | 41x   |
| 6     | 6             | ``6.94\times10^{-3}`` | 64                | ``6.10\times10^{-5}`` | 114x  |
| 7     | 7             | ``5.10\times10^{-3}`` | 128               | ``1.53\times10^{-5}`` | 334x  |
| 8     | 8             | ``3.91\times10^{-3}`` | 256               | ``3.81\times10^{-6}`` | 1024x |

SOS2 error decays **polynomially** as ``O(1/L^2)``; sawtooth error decays
**exponentially** as ``O(4^{-L})``.

### Same number of segments

When both methods use the same number of uniform segments ``n``, they produce identical
PWL interpolations, so the approximation error is the same. The difference is in
formulation size: SOS2 needs ``O(n)`` binary variables, sawtooth needs only
``\log_2(n)``.

| Segments ``n`` | Error                 | SOS2 vars | Sawtooth vars | Var ratio |
|:-------------- |:--------------------- |:--------- |:------------- |:--------- |
| 2              | ``6.25\times10^{-2}`` | 1         | 1             | 1.0x      |
| 4              | ``1.56\times10^{-2}`` | 3         | 2             | 1.5x      |
| 8              | ``3.91\times10^{-3}`` | 7         | 3             | 2.3x      |
| 16             | ``9.77\times10^{-4}`` | 15        | 4             | 3.8x      |
| 32             | ``2.44\times10^{-4}`` | 31        | 5             | 6.2x      |
| 64             | ``6.10\times10^{-5}`` | 63        | 6             | 10.5x     |
| 128            | ``1.53\times10^{-5}`` | 127       | 7             | 18.1x     |
| 256            | ``3.81\times10^{-6}`` | 255       | 8             | 31.9x     |

## Extension to Bilinear Terms

Bilinear terms ``z = xy`` arise throughout optimization (energy systems, pooling problems,
gas networks). The standard univariate reformulation uses the identity:

```math
xy = \frac{1}{4}\left[(x + y)^2 - (x - y)^2\right]
```

Each squared term is approximated independently with one of the methods above. With both
terms at the same refinement level, the bilinear error satisfies:

```math
\varepsilon_{xy} \leq \frac{1}{2} \, \varepsilon_{\text{quad}}
```

All scaling relationships from the univariate case carry over with a constant factor of
``1/2``. The exponential gap between SOS2 and sawtooth persists.

## Practical Considerations

**SOS2** has the advantage of being natively supported by commercial solvers (Gurobi,
CPLEX) with specialized branching rules. Both the solver SOS2 and manual SOS2
formulations produce sharp LP relaxations for convex functions.

**Sawtooth** introduces auxiliary continuous variables and big-M-type constraints, which
may interact less favorably with presolve and cutting planes. However, Beach et al. (2023)
show that the sawtooth relaxation is **sharp** (its LP relaxation equals the convex hull)
and **hereditarily sharp** (sharpness is preserved at every node in the branch-and-bound
tree). This strong theoretical property, combined with exponentially fewer binary
variables, can lead to significant solver performance gains for problems requiring high
approximation accuracy.

Whether the tighter approximation at a given variable budget outweighs the structural
advantages of SOS2 depends on the specific problem and solver.

## References

  - Beach, B., Burlacu, R., Barmann, A., Hager, L., Kleinert, T. (2023). *Enhancements of discretization approaches for non-convex MIQCQPs*. Journal of Global Optimization.
  - Barmann, A., Burlacu, R., Hager, L., Kleinert, T. (2023). *On piecewise linear approximations of bilinear terms: structural comparison of univariate and bivariate MIP formulations*. Journal of Global Optimization, 85, 789-819.
  - Yarotsky, D. (2017). *Error bounds for approximations with deep ReLU networks*. Neural Networks, 94, 103-114.
  - Huchette, J.A. (2018). *Advanced mixed-integer programming formulations: methodology, computation, and application*. PhD thesis, MIT.
