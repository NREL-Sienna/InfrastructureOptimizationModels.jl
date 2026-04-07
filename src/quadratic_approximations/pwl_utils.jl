# Shared utilities for piecewise linear approximation methods.

"""
    _get_breakpoints_for_pwl_function(min_val, max_val, f; num_segments = DEFAULT_INTERPOLATION_LENGTH)

Generate breakpoints for piecewise linear (PWL) approximation of a nonlinear function.

This function creates equally-spaced breakpoints over the specified domain [min_val, max_val]
and evaluates the given function at each breakpoint to construct a piecewise linear approximation.
The breakpoints are used in optimization problems to linearize nonlinear constraints or objectives.

# Arguments
- `min_val::Float64`: Minimum value of the domain for the PWL approximation
- `max_val::Float64`: Maximum value of the domain for the PWL approximation
- `f`: Function to be approximated (must be callable with Float64 input)
- `num_segments::Int`: Number of linear segments in the PWL approximation (default: DEFAULT_INTERPOLATION_LENGTH)

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: A tuple containing:
  - `x_bkpts`: Vector of x-coordinates (breakpoints) in the domain
  - `y_bkpts`: Vector of y-coordinates (function values at breakpoints)

# Notes
- The number of breakpoints is `num_segments + 1`
- Breakpoints are equally spaced across the domain
- The first breakpoint is always at `min_val` and the last at `max_val`
"""
function _get_breakpoints_for_pwl_function(
    min_val::Float64,
    max_val::Float64,
    f;
    num_segments = DEFAULT_INTERPOLATION_LENGTH,
)
    # Calculate total number of breakpoints (one more than segments)
    # num_segments is the number of linear segments in the PWL approximation
    # num_bkpts is the total number of breakpoints needed for the segments
    num_bkpts = num_segments + 1

    # Calculate step size for equally-spaced breakpoints
    step = (max_val - min_val) / num_segments

    # Pre-allocate vectors for breakpoint coordinates
    x_bkpts = Vector{Float64}(undef, num_bkpts)  # Domain values (x-coordinates)
    y_bkpts = Vector{Float64}(undef, num_bkpts)  # Function values (y-coordinates)

    # Set the first breakpoint at the minimum domain value
    x_bkpts[1] = min_val
    y_bkpts[1] = f(min_val)

    # Generate remaining breakpoints by stepping through the domain
    for i in 1:num_segments
        x_val = min_val + step * i  # Calculate x-coordinate of current breakpoint
        x_bkpts[i + 1] = x_val
        y_bkpts[i + 1] = f(x_val)  # Evaluate function at current breakpoint
    end
    return x_bkpts, y_bkpts
end

"Helper: returns x² (used as the default function for PWL breakpoint generation)."
_square(x::Float64) = x * x
