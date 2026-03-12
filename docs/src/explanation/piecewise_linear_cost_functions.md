# Piecewise Linear Cost Functions

```@meta
CurrentModule = InfrastructureOptimizationModels
```

This package provides two formulations for representing piecewise linear (PWL) cost
functions in optimization models. Both formulations express an operating point and its
cost as a point on a piecewise linear curve connecting breakpoints
``(P_0, C_0), (P_1, C_1), \ldots, (P_n, C_n)``, but they approach the problem differently.

## Lambda Formulation (Convex Combination)

Used by `CostCurve{PiecewisePointCurve}` and `FuelCurve{PiecewisePointCurve}`.

The lambda formulation assigns a weight ``\lambda_i`` to each **breakpoint**. The
operating point and cost are expressed as a weighted average of the breakpoint values.

### Formulation

Given ``n + 1`` breakpoints with power levels ``P_0, \ldots, P_n`` and costs
``C(P_0), \ldots, C(P_n)``:

**Variables:**

```math
\lambda_i \in [0, 1], \quad i = 0, 1, \ldots, n
```

**Constraints:**

```math
\begin{aligned}
p &= \sum_{i=0}^{n} \lambda_i \, P_i && \text{(linking)} \\
C(p) &= \sum_{i=0}^{n} \lambda_i \, C(P_i) && \text{(cost)} \\
\sum_{i=0}^{n} \lambda_i &= u && \text{(normalization, } u \text{ = on-status)} \\
\text{at most two adjacent } &\lambda_i \text{ may be nonzero} && \text{(adjacency)}
\end{aligned}
```

### Adjacency Enforcement

The adjacency condition keeps the operating point on a single line segment. For
**convex** cost curves (where each segment is more expensive than the last), the
optimizer naturally satisfies this condition — no extra constraints are needed.

For **non-convex** cost curves, an
[SOS2 constraint](https://en.wikipedia.org/wiki/Special_ordered_set) is added to
enforce adjacency explicitly. This requires solver support for SOS2, and effectively
introduces additional branching in the solver.

### Compact Form

When the power variable represents output above minimum generation
(PowerAboveMinimumVariable), a compact variant adjusts the linking constraint to
include a ``P_{\min}`` offset:

```math
\sum_{i=0}^{n} \lambda_i \, P_i = p + P_{\min} \cdot u
```

### Variables and Constraints

| Container Type                                       | Description                                  |
|:---------------------------------------------------- |:-------------------------------------------- |
| [`PiecewiseLinearCostVariable`](@ref)                | Lambda weights ``\lambda_i \in [0, 1]``      |
| [`PiecewiseLinearCostConstraint`](@ref)              | Links power variable to weighted breakpoints |
| [`PiecewiseLinearCostNormalizationConstraint`](@ref) | Ensures ``\sum \lambda_i = u``               |

### Implementation

| Function                        | Role                                                           |
|:------------------------------- |:-------------------------------------------------------------- |
| `_add_pwl_variables!`           | Creates ``n + 1`` lambda variables per time step               |
| `_add_pwl_constraint_standard!` | Adds linking and normalization constraints                     |
| `_add_pwl_constraint_compact!`  | Compact form with ``P_{\min}`` offset                          |
| `_add_pwl_term!`                | Orchestrates variables, constraints, SOS2, and cost expression |
| `add_pwl_sos2_constraint!`      | Adds SOS2 adjacency for non-convex curves                      |

## Delta Formulation (Incremental / Block Offers)

Used by `MarketBidCost` and `ImportExportCost` offer curves.

The delta formulation assigns a variable ``\delta_k`` to each **segment**, representing
how much of that segment has been used. The operating point is the sum of segment
contributions from the first breakpoint.

### Formulation

Given ``n`` segments with breakpoints ``P_0, \ldots, P_n`` and marginal costs (slopes)
``m_k = \frac{C(P_k) - C(P_{k-1})}{P_k - P_{k-1}}``:

**Variables:**

```math
\delta_k \geq 0, \quad k = 1, \ldots, n
```

**Constraints:**

```math
\begin{aligned}
p &= \sum_{k=1}^{n} \delta_k + P_{\min,\text{offset}} && \text{(linking)} \\
\delta_k &\leq P_k - P_{k-1} && \text{(segment capacity)} \\
C(p) &= \sum_{k=1}^{n} m_k \, \delta_k && \text{(cost)}
\end{aligned}
```

### Convexity Advantage

For convex offer curves (``m_1 \leq m_2 \leq \cdots \leq m_n``), the fill-order
condition is automatically satisfied. A cost-minimizing optimizer fills cheap segments
before expensive ones, so no SOS2 constraints or binary variables are needed. This is
the common case in power systems, where generator marginal costs are typically
increasing.

### Variables and Constraints

| Container Type                                           | Description                            |
|:-------------------------------------------------------- |:-------------------------------------- |
| [`PiecewiseLinearBlockIncrementalOffer`](@ref)           | Delta variables for incremental offers |
| [`PiecewiseLinearBlockDecrementalOffer`](@ref)           | Delta variables for decremental offers |
| [`PiecewiseLinearBlockIncrementalOfferConstraint`](@ref) | Linking + capacity for incremental     |
| [`PiecewiseLinearBlockDecrementalOfferConstraint`](@ref) | Linking + capacity for decremental     |

### Implementation

| Function                           | Role                                                     |
|:---------------------------------- |:-------------------------------------------------------- |
| `add_pwl_variables!`               | Creates ``n`` delta variables per time step              |
| `_add_pwl_constraint!`             | Adds linking and segment capacity constraints            |
| `add_pwl_term!`                    | Orchestrates variables, constraints, and cost expression |
| `get_pwl_cost_expression`          | Builds ``\sum m_k \, \delta_k`` cost expression          |
| `add_pwl_block_offer_constraints!` | Low-level block-offer constraint builder                 |

## Comparison

|                      | Lambda (``\lambda``)               | Delta (``\delta``)                  |
|:-------------------- |:---------------------------------- |:----------------------------------- |
| **Thinks about**     | Breakpoints (the dots)             | Segments (the lines)                |
| **Variables**        | ``n + 1`` (one per breakpoint)     | ``n`` (one per segment)             |
| **Output equation**  | ``p = \sum \lambda_i \, P_i``      | ``p = \sum \delta_k + P_{\min}``    |
| **Cost equation**    | ``C = \sum \lambda_i \, C(P_i)``   | ``C = \sum m_k \, \delta_k``        |
| **Adjacency rule**   | Must be enforced explicitly (SOS2) | Often automatic (convex case)       |
| **Binary variables** | Usually needed (non-convex)        | Often not needed (convex case)      |
| **Used by**          | `CostCurve`, `FuelCurve`           | `MarketBidCost`, `ImportExportCost` |

## When Does the Choice Matter?

For **convex** cost functions — where each segment costs more than the last — the delta
formulation is typically simpler and faster. The optimizer fills cheap segments first on
its own, so no extra adjacency constraints are needed. This is the common case in power
systems.

For **non-convex** cost functions — where a segment might be cheaper than the one before
it — both formulations need additional constraints (SOS2 or binary variables) to force
the correct ordering. In that case the lambda formulation is the traditional choice.

## Incremental Interpolation (General PWL Approximation)

In addition to cost functions, the package provides a general-purpose incremental
(delta) method for approximating arbitrary nonlinear functions as piecewise linear. This
is used for PWL approximation of constraints (e.g., loss functions, quadratic terms).

### Formulation

Given breakpoints ``(x_1, y_1), \ldots, (x_{n+1}, y_{n+1})`` where ``y_i = f(x_i)``,
and interpolation variables ``\delta_i \in [0, 1]`` with binary ordering variables
``z_i \in \{0, 1\}``:

```math
\begin{aligned}
x &= x_1 + \sum_{i=1}^{n} \delta_i \, (x_{i+1} - x_i) \\
y &= y_1 + \sum_{i=1}^{n} \delta_i \, (y_{i+1} - y_i) \\
z_i &\geq \delta_{i+1}, \quad z_i \leq \delta_i \quad \text{(ordering)}
\end{aligned}
```

The ordering constraints ensure segments are filled sequentially: ``\delta_1`` must be
filled before ``\delta_2`` can begin.

### Implementation

| Function                                             | Role                                        |
|:---------------------------------------------------- |:------------------------------------------- |
| `_get_breakpoints_for_pwl_function`                  | Generates equally-spaced breakpoints        |
| `add_sparse_pwl_interpolation_variables!`            | Creates ``\delta`` and ``z`` variables      |
| `_add_generic_incremental_interpolation_constraint!` | Adds interpolation and ordering constraints |
