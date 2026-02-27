# InfrastructureOptimizationModels.jl

```@meta
CurrentModule = InfrastructureOptimizationModels
```

## Overview

`InfrastructureOptimizationModels.jl` is a [`Julia`](http://www.julialang.org) package that provides core abstractions and optimization model structures for power systems operations modeling. It defines `DecisionModel` and `EmulationModel` types along with their associated optimization containers, formulations, and output handling capabilities.

## About

`InfrastructureOptimizationModels` is part of the National Lab of the Rockies NLR (formerly known as NREL)
[Sienna ecosystem](https://www.nrel.gov/analysis/sienna.html), an open source framework for
scheduling problems and dynamic simulations for power systems. The Sienna ecosystem can be
[found on github](https://github.com/NREL-Sienna/Sienna). It contains three applications:

  - [Sienna\Data](https://github.com/NREL-Sienna/Sienna?tab=readme-ov-file#siennadata) enables
    efficient data input, analysis, and transformation
  - [Sienna\Ops](https://github.com/NREL-Sienna/Sienna?tab=readme-ov-file#siennaops) enables
    enables system scheduling simulations by formulating and solving optimization problems
  - [Sienna\Dyn](https://github.com/NREL-Sienna/Sienna?tab=readme-ov-file#siennadyn) enables
    system transient analysis including small signal stability and full system dynamic
    simulations

Each application uses multiple packages in the [`Julia`](http://www.julialang.org)
programming language.
