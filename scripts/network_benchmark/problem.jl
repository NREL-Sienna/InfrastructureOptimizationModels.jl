using Random

# ─── Network data ────────────────────────────────────────────────────────────

abstract type AbstractNetworkProblem end

struct LosslessNetworkProblem <: AbstractNetworkProblem
    N::Int
    K::Int
    gen_nodes::Vector{String}
    dem_nodes::Vector{String}
    all_nodes::Vector{String}
    edges::Vector{Tuple{String, String}}
    conductances::Dict{Tuple{String, String}, Float64}
    demands::Dict{String, Float64}
    marginal_costs::Dict{String, Vector{Float64}}
    segment_widths::Dict{String, Vector{Float64}}
end

struct LossyNetworkProblem <: AbstractNetworkProblem
    N::Int
    K::Int
    gen_nodes::Vector{String}
    dem_nodes::Vector{String}
    all_nodes::Vector{String}
    edges::Vector{Tuple{String, String}}
    conductances::Dict{Tuple{String, String}, Float64}
    demands::Dict{String, Float64}
    marginal_costs::Dict{String, Vector{Float64}}
    segment_widths::Dict{String, Vector{Float64}}
    loss_coeffs::Dict{String, Vector{Float64}}
end

"""
Generate a random connected network matching the model in bilinear_delta_model.tex.

- `N` nodes split evenly into generators and demands.
- `K` PWL cost segments per generator with nondecreasing marginal costs.
- Connected graph via random spanning tree plus extra edges.
- Conductances scaled for feasibility given voltage/current bounds.
- Demands d_i ∈ [0.05, 0.15].
"""
function generate_lossy_network(; N::Int = 10, K::Int = 3, seed::Int = 42)
    @assert iseven(N) "N must be even"
    rng = MersenneTwister(seed)

    all_nodes = ["n$(i)" for i in 1:N]
    gen_nodes = all_nodes[1:(N ÷ 2)]
    dem_nodes = all_nodes[(N ÷ 2 + 1):N]

    edges = Tuple{String, String}[]
    conductances = Dict{Tuple{String, String}, Float64}()

    # Random spanning tree: shuffle nodes, connect consecutive pairs
    perm = shuffle(rng, 1:N)
    for idx in 1:(N - 1)
        a, b = all_nodes[perm[idx]], all_nodes[perm[idx + 1]]
        e = a < b ? (a, b) : (b, a)
        if e ∉ edges
            push!(edges, e)
            conductances[e] = 1.0 + 4.0 * rand(rng)
        end
    end

    # Extra edges for density
    for _ in 1:(N ÷ 2)
        i, j = rand(rng, 1:N), rand(rng, 1:N)
        i == j && continue
        a, b = all_nodes[i], all_nodes[j]
        e = a < b ? (a, b) : (b, a)
        if e ∉ edges
            push!(edges, e)
            conductances[e] = 1.0 + 4.0 * rand(rng)
        end
    end

    # Demands at demand nodes (small enough for network to transport)
    demands = Dict(d => 0.05 + 0.1 * rand(rng) for d in dem_nodes)

    # PWL cost data: K segments per generator with nondecreasing marginal costs
    marginal_costs = Dict{String, Vector{Float64}}()
    segment_widths = Dict{String, Vector{Float64}}()
    for g in gen_nodes
        mc = sort(rand(rng, K) .* 10.0 .+ 1.0)
        marginal_costs[g] = mc
        widths = rand(rng, K) .+ 0.1
        widths .*= 1.5 / sum(widths)
        segment_widths[g] = widths
    end

    loss_coeffs = Dict{String, Vector{Float64}}()
    for g in gen_nodes
        loss_coeffs[g] = rand(rng, 3) * 0.001
    end

    return LossyNetworkProblem(
        N, K, gen_nodes, dem_nodes, all_nodes,
        edges, conductances, demands,
        marginal_costs, segment_widths,
        loss_coeffs
    )
end

function generate_lossless_network(; N::Int = 10, K::Int = 3, seed::Int = 42)
    net = generate_lossy_network(; N, K, seed)

    return LosslessNetworkProblem(
        N, K, net.gen_nodes, net.dem_nodes, net.all_nodes,
        net.edges, net.conductances, net.demands,
        net.marginal_costs, net.segment_widths,
    )
end

"""Build adjacency list from edge set."""
function adjacency_list(net::AbstractNetworkProblem)
    adj = Dict{String, Vector{Tuple{String, Float64}}}()
    for n in net.all_nodes
        adj[n] = Tuple{String, Float64}[]
    end
    for (a, b) in net.edges
        g = net.conductances[(a, b)]
        push!(adj[a], (b, g))
        push!(adj[b], (a, g))
    end
    return adj
end
