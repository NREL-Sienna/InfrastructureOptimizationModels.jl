using Documenter
import DataStructures: OrderedDict
using InfrastructureOptimizationModels
using DocumenterInterLinks

links = InterLinks(
    "PowerSystems" => "https://nrel-sienna.github.io/PowerSystems.jl/stable/",
    "PowerSimulations" => "https://nrel-sienna.github.io/PowerSimulations.jl/stable/",
)

include(joinpath(@__DIR__, "make_tutorials.jl"))
make_tutorials()

pages = OrderedDict(
    "Welcome Page" => "index.md",
    # "Tutorials" => Any["stub" => "tutorials/generated_stub.md"],
    # "How to..." => Any["stub" => "how_to_guides/stub.md"],
    # "Explanation" => Any["stub" => "explanation/stub.md"],
    "Reference" => Any[
        "Developers" => ["Developer Guidelines" => "reference/developer_guidelines.md",
        "Internals" => "reference/internal.md"],
        "Public API" => "reference/public.md",
    ],
)

makedocs(
    modules = [InfrastructureOptimizationModels],
    format = Documenter.HTML(
        prettyurls = haskey(ENV, "GITHUB_ACTIONS"),
        size_threshold = nothing,),
    sitename = "github.com/NREL-Sienna/InfrastructureOptimizationModels.jl",
    authors = "NREL-Sienna",
    pages = Any[p for p in pages],
    draft = false,
    plugins = [links],
)

deploydocs(
    repo="github.com/NREL-Sienna/InfrastructureOptimizationModels.jl",
    target="build",
    branch="gh-pages",
    devbranch="main",
    devurl="dev",
    push_preview=true,
    versions=["stable" => "v^", "v#.#"],
)
