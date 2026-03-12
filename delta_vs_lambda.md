# The Lambda and Delta Formulations

*A Kid-Friendly Guide to Piecewise Linear Cost Functions*

---

## What Are We Trying to Do?

Imagine you run a lemonade stand. Making the first 10 cups is cheap because you already have the supplies. But the next 10 cups cost more because you need to run to the store. And the last 10 cups? Really expensive—you have to buy a whole new pitcher.

This is a **cost function**: the price of making lemonade goes up in steps. In the real world, power plants, factories, and all sorts of things have costs that work this way. We draw these costs as a curve, but curves are hard for computers. So we chop the curve into straight-line pieces. That's a **piecewise linear approximation**.

The question is: how do we tell a computer where we are on those line segments? There are two popular ways—the **lambda** formulation and the **delta** formulation. They solve the same problem, but they think about it differently.

Suppose we have breakpoints $x_0, x_1, \ldots, x_n$ with corresponding costs $C(x_0), C(x_1), \ldots, C(x_n)$. Both formulations need to express an output level $x$ and its cost $C(x)$ as a point on the piecewise linear curve connecting these breakpoints.

## The Lambda ($\lambda$) Formulation

### The Idea

Think of the breakpoints on your cost curve as dots on a page. The lambda method says: "I can describe any point on the curve by mixing these dots together."

You assign a weight ($\lambda$) to each breakpoint. These weights must add up to 1, and each weight is between 0 and 1. Your output and your cost are just the weighted average of the breakpoint values.

### The Equations

We introduce one weight $\lambda_i$ for each breakpoint $i = 0, 1, \ldots, n$. The formulation is:

$$
x = \sum_{i=0}^{n} \lambda_i \, x_i \qquad \text{(output)}
$$

$$
C(x) = \sum_{i=0}^{n} \lambda_i \, C(x_i) \qquad \text{(cost)}
$$

$$
\sum_{i=0}^{n} \lambda_i = 1 \qquad \text{(weights sum to 1)}
$$

$$
\lambda_i \geq 0 \quad \forall \, i \qquad \text{(non-negativity)}
$$

$$
\text{at most two adjacent } \lambda_i \text{ may be nonzero} \qquad \text{(SOS2 / adjacency)}
$$

In plain language: your output $x$ is a weighted mix of the breakpoint locations, and your cost $C(x)$ is the same mix of the breakpoint costs. The adjacency rule (last line) is the tricky part—it keeps you on a single line segment instead of jumping across the curve.

### An Example

Say your lemonade stand has three breakpoints: 0 cups (\$0), 10 cups (\$5), and 20 cups (\$15). If you set $\lambda_1 = 0.5$ and $\lambda_2 = 0.5$, you're exactly halfway between the first and second dot: 5 cups at \$2.50.

### The Catch

There's a rule: at most two neighboring breakpoints can have nonzero weights. You can't mix dot 1 and dot 3 while skipping dot 2. This is called an **adjacency condition**, and it's what keeps you on the actual curve instead of cutting corners. Enforcing this condition is the main headache of the lambda method—it typically requires special ordered sets (SOS2 constraints) or extra binary variables.

| Feature | Description |
|---|---|
| **Variables** | One weight $\lambda_i$ per breakpoint ($n + 1$ variables) |
| **Key constraint** | Weights sum to 1; adjacency (SOS2) required |
| **How it picks a point** | Weighted average of breakpoint values |
| **Main difficulty** | Enforcing the adjacency condition |

## The Delta ($\delta$) Formulation

### The Idea

The delta method thinks about the segments, not the dots. Instead of asking "where am I between breakpoints?" it asks "how much of each segment have I used up?"

Each segment gets a variable $\delta$ that tells you how far along that segment you've gone, from 0 (haven't started it) to its maximum length (used it all up). Your total output is the starting point plus the sum of all the $\delta$ values.

### The Equations

We introduce one variable $\delta_i$ for each segment $i = 1, 2, \ldots, n$, and define the slope of each segment as:

$$
m_i = \frac{C(x_i) - C(x_{i-1})}{x_i - x_{i-1}}
$$

The formulation is:

$$
x = x_0 + \sum_{i=1}^{n} \delta_i \qquad \text{(output)}
$$

$$
C(x) = C(x_0) + \sum_{i=1}^{n} m_i \, \delta_i \qquad \text{(cost)}
$$

$$
0 \leq \delta_i \leq (x_i - x_{i-1}) \quad \forall \, i \qquad \text{(bounded by segment length)}
$$

$$
\text{If convex } (m_1 \leq m_2 \leq \cdots \leq m_n): \text{ no extra constraints needed}
$$

$$
\text{If non-convex: } \delta_{i+1} > 0 \implies \delta_i = (x_i - x_{i-1}) \qquad \text{(fill-order, needs binaries)}
$$

In plain language: your output $x$ starts at the first breakpoint and adds up contributions from each segment. The cost does the same thing, but weights each segment's contribution by its slope $m_i$. If costs are convex, the optimizer fills cheap segments first on its own—no adjacency trick required.

### An Example

Back to lemonade. Segment 1 covers 0–10 cups, segment 2 covers 10–20 cups. If you've made 15 cups, then $\delta_1 = 10$ (segment 1 is full) and $\delta_2 = 5$ (you're halfway through segment 2). Total output: $0 + 10 + 5 = 15$. The cost is the slope of each segment times the amount you used.

### The Nice Part

The filling-order rule—you must fill segment 1 before you start segment 2—often enforces itself automatically when costs are **convex** (each segment is more expensive than the last). The optimizer naturally fills cheap segments first. That means you don't always need SOS2 constraints or extra binary variables, which makes the problem easier to solve.

| Feature | Description |
|---|---|
| **Variables** | One variable $\delta_i$ per segment ($n$ variables) |
| **Key constraint** | Each $\delta_i$ bounded by segment length |
| **How it picks a point** | Sum of segment contributions from $x_0$ |
| **Main advantage** | Convexity enforces fill order; fewer binaries needed |

## How They Compare

| | Lambda ($\lambda$) | Delta ($\delta$) |
|---|---|---|
| **Thinks about…** | Breakpoints (the dots) | Segments (the lines) |
| **Variables** | $n + 1$ (one per breakpoint) | $n$ (one per segment) |
| **Output equation** | $x = \sum \lambda_i x_i$ | $x = x_0 + \sum \delta_i$ |
| **Cost equation** | $C = \sum \lambda_i C(x_i)$ | $C = C(x_0) + \sum m_i \delta_i$ |
| **Adjacency rule** | Must be enforced explicitly | Often automatic (convex case) |
| **Binary variables** | Usually needed | Often not needed (convex case) |
| **Non-convex costs** | Works (with binaries) | Needs extra binaries too |
| **Intuition** | "Mix neighboring dots" | "Fill segments in order" |

## When Does the Choice Matter?

For convex cost functions—where each segment costs more than the last—the delta formulation is typically simpler and faster. The optimizer fills cheap segments first on its own, so you don't need extra constraints. This comes up a lot in power systems, where generator cost curves are almost always convex.

For non-convex cost functions—where a segment might be cheaper than the one before it—both formulations need binary variables to force the correct ordering. In that case, the lambda formulation is the more traditional choice, though either can work.

The bottom line: if your costs go up with output (convex), delta tends to give you a tighter, easier problem. If they don't, the two methods are roughly comparable in difficulty.

## The Takeaway

Lambda says: "I'm a mix of two neighboring dots." Delta says: "I've filled up this much of each segment." Both get you to the same place on the cost curve. Delta is often the easier route when costs are convex, because the fill-in-order rule takes care of itself. Lambda is the classic approach and works for any shape, but needs more help from the solver to stay on the curve.
