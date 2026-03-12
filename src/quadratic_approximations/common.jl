"""
Convenience macro for quadratic approximations.

Specify sparse by putting :sparse in the last two args
Specify custom meta by putting meta_-prefixed variable in the last two args
"""

macro _add_container!(type, key, args...)
    fname = Symbol("add_", type, "_container!")

    offset = 0
    sparse = :(false)
    meta = :(meta)
    if !isempty(args)
        start = max(length(args) - 1, 1)
        for arg in args[start:end]
            if arg == :sparse
                sparse = :(true)
                offset += 1
            end
            if occursin("meta_", string(arg))
                meta = arg
                offset += 1
            end
        end
    end

    axes = args[1:(end - offset)]
    return esc(
        :(
            $fname(
            container,
            $key(),
            C,
            names,
            $(axes...),
            time_steps;
            meta = $meta,
            sparse = $sparse,
        )
        ),
    )
end
