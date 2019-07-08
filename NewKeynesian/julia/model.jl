#%% cell
#=

## Types

First we have the `Params` type, which holds all the model parameters as well
as the paramters that drive the algorithm.
=#

#%% cell
@with_kw struct Params
    zlb::Bool = true
    γ::Float64 = 1.0       # Utility-function parameter
    β::Float64 = 0.99      # Discount factor
    ϑ::Float64 = 2.09      # Utility-function parameter
    ϵ::Float64 = 4.45      # Parameter in the Dixit-Stiglitz aggregator
    ϕ_y::Float64 = 0.07    # Parameter of the Taylor rule
    ϕ_π::Float64 = 2.21    # Parameter of the Taylor rule
    μ::Float64 = 0.82      # Parameter of the Taylor rule
    Θ::Float64 = 0.83      # Share of non-reoptimizing firms (Calvo's pricing)
    πstar::Float64 = 1.0   # Target (gross) inflation rate
    gbar::Float64 = 0.23   # Steady-state share of government spending in output

    # autocorrelation coefficients
    ρηR::Float64 = 0.0   # See process (28) in MM (2015)
    ρηa::Float64 = 0.95  # See process (22) in MM (2015)
    ρηL::Float64 = 0.25  # See process (16) in MM (2015)
    ρηu::Float64 = 0.92  # See process (15) in MM (2015)
    ρηB::Float64 = 0.0   # See process (17) in MM (2015)
    ρηG::Float64 = 0.95  # See process (26) in MM (2015)

    # standard deviations
    σηR::Float64 = 0.0028  # See process (28) in MM (2015)
    σηa::Float64 = 0.0045  # See process (22) in MM (2015)
    σηL::Float64 = 0.05    # See process (16) in MM (2015)
    σηu::Float64 = 0.0054  # See process (15) in MM (2015)
    σηB::Float64 = 0.001   # See process (17) in MM (2015)
    σηG::Float64 = 0.0038  # See process (26) in MM (2015)

    # algorithm parameters
    deg::Int = 2          # max degree of complete monomial
    damp::Float64 = 0.1   # dampening parameter for coefficient update
    tol::Float64 = 1e-7   # Tolerance for stopping iterations
    grid_kind::Symbol = :sobol  # type of grid (:sobol or :random)
end

function grid_size(p::Params)
    Dict(1 => 20, 2 => 100, 3 => 300, 4 => 1000, 5 => 2000)[p.deg]
end

# returns the covariance matrix for the 6 shocks in the model
vcov(p::Params) = diagm(0 => [p.σηR^2, p.σηa^2, p.σηL^2, p.σηu^2, p.σηB^2, p.σηG^2])

#%% cell
#=
Next we have a type called `SteadyState` that is intended to hold the
deterministic steady state realization for each variable in the model.
=#

#%% cell
struct SteadyState
    Yn::Float64
    Y::Float64
    π::Float64
    δ::Float64
    L::Float64
    C::Float64
    F::Float64
    S::Float64
    R::Float64
    w::Float64
end

function SteadyState(p::Params)
    Yn_ss = exp(p.gbar)^(p.γ/(p.ϑ+p.γ))
    Y_ss  = Yn_ss
    π_ss  = 1.0
    δ_ss  = 1.0
    L_ss  = Y_ss/δ_ss
    C_ss  = (1-p.gbar)*Y_ss
    F_ss  = C_ss^(-p.γ)*Y_ss/(1-p.β*p.Θ*π_ss^(p.ϵ-1))
    S_ss  = L_ss^p.ϑ*Y_ss/(1-p.β*p.Θ*π_ss^p.ϵ)
    R_ss  = π_ss/p.β
    w_ss  = (L_ss^p.ϑ)*(C_ss^p.γ)

    SteadyState(Yn_ss, Y_ss, π_ss, δ_ss, L_ss, C_ss, F_ss, S_ss, R_ss, w_ss)
end

#%% cell
#=
Given an instance of `Params` and `SteadyState`, we can construct the grid on
which we will solve the model.

The `Grids` type holds this grid as well as matrices used to compute
expectations.

To match the Python and Matlab versions of the code, the constructor for `Grids`
below loads pre-generated grids from a `.mat` file for both Sobol and random
grids. This ensures that the exact same code is run in each language. If you
would like to generate the grids in pure Julia, you can set the `grid_source`
keyword argument to `:julia`
=#

#%% cell
struct Grids
    # period t grids
    ηR::Vector{Float64}
    ηa::Vector{Float64}
    ηL::Vector{Float64}
    ηu::Vector{Float64}
    ηB::Vector{Float64}
    ηG::Vector{Float64}
    R::Vector{Float64}
    δ::Vector{Float64}

    # combined matrix and complete polynomial version of it
    X::Matrix{Float64}
    X0_G::Dict{Int,Matrix{Float64}}

    # quadrature weights and nodes
    ϵ_nodes::Matrix{Float64}
    ω_nodes::Vector{Float64}

    # period t+1 grids at all shocks
    ηR1::Matrix{Float64}
    ηa1::Matrix{Float64}
    ηL1::Matrix{Float64}
    ηu1::Matrix{Float64}
    ηB1::Matrix{Float64}
    ηG1::Matrix{Float64}
end

function Grids(p::Params, ss::SteadyState; grid_source::Symbol=:mat)
    m = grid_size(p)
    σ = [p.σηR p.σηa p.σηL p.σηu p.σηB p.σηG]
    ρ = [p.ρηR p.ρηa p.ρηL p.ρηu p.ρηB p.ρηG]
    if p.grid_kind == :sobol
        if grid_source == :mat
            # Values of exogenous state variables are in the interval +/- σ/sqrt(1-ρ^2)
            _path = joinpath(@__DIR__, "..", "Sobol_grids.mat")
            s = (matread(_path)["Sobol_grids"][1:m, :])::Matrix{Float64}
            sη = s[:, 1:6]
            η = (
                -2σ .+ 4.0 .* (maximum(sη, dims=1).-sη) ./
                (maximum(sη, dims=1).-minimum(sη, dims=1)).*σ
            )./sqrt.(1 .- ρ.^2)

            R = 1 .+ 0.05 .* (maximum(s[:, 7]).-s[:, 7]) ./ (maximum(s[:, 7])-minimum(s[:, 7]))
            δ = 0.95 .+ 0.05 .* (maximum(s[:, 8]).-s[:, 8]) ./ (maximum(s[:, 8])-minimum(s[:, 8]))
        else
            ub = [2 * p.σηR / sqrt(1 - p.ρηR^2),
                  2 * p.σηa / sqrt(1 - p.ρηa^2),
                  2 * p.σηL / sqrt(1 - p.ρηL^2),
                  2 * p.σηu / sqrt(1 - p.ρηu^2),
                  2 * p.σηB / sqrt(1 - p.ρηB^2),
                  2 * p.σηG / sqrt(1 - p.ρηG^2),
                  1.05,  # R
                  1.0    # δ
                 ]
            lb = -ub
            lb[[7, 8]] = [1.0, 0.95]  # adjust lower bound for R and δ

            # construct SobolSeq
            s = SobolSeq(length(ub), ub, lb)
            # skip(s, m)  # See note in README of Sobol.jl

            # gather points
            seq = Array{Float64}(8, m)
            for i in 1:m
                seq[:, i] = next(s)
            end
            seq = seq'  # transpose so variables are in columns
            η = seq[:, 1:6]
            R  = seq[:, 7]
            δ  = seq[:, 8]
        end
    else  # assume random
        # Values of exogenous state variables are distributed uniformly
        # in the interval +/- std/sqrt(1-rho_nu^2)
        if grid_source == :mat
            _path = joinpath(@__DIR__, "..", "random_grids.mat")
            s = (matread(_path)["random_grids"][1:m, :])::Matrix{Float64}
        else
            s = rand(m, 8)
        end
        sη = s[:, 1:6]
        η = @. (-2*σ + 4*σ*sη) / sqrt(1-ρ^2)

        # Values of endogenous state variables are distributed uniformly
        # in the intervals [1 1.05] and [0.95 1], respectively
        R = 1 + 0.05 * s[:, 7]
        δ = 0.95 + 0.05 * s[:, 8]
    end
    ηR = η[:, 1]
    ηa = η[:, 2]
    ηL = η[:, 3]
    ηu = η[:, 4]
    ηB = η[:, 5]
    ηG = η[:, 6]

    X = [log.(R) log.(δ) ηR ηa ηL ηu ηB ηG]
    X0_G = Dict(
        1 => complete_polynomial(X, 1),
        p.deg => complete_polynomial(X, p.deg)
    )

    ϵ_nodes, ω_nodes = qnwmonomial1(vcov(p))

    ηR1 = p.ρηR.*ηR .+ ϵ_nodes[:, 1]'
    ηa1 = p.ρηa.*ηa .+ ϵ_nodes[:, 2]'
    ηL1 = p.ρηL.*ηL .+ ϵ_nodes[:, 3]'
    ηu1 = p.ρηu.*ηu .+ ϵ_nodes[:, 4]'
    ηB1 = p.ρηB.*ηB .+ ϵ_nodes[:, 5]'
    ηG1 = p.ρηG.*ηG .+ ϵ_nodes[:, 6]'

    Grids(
        ηR, ηa, ηL, ηu, ηB, ηG, R, δ, X, X0_G, ϵ_nodes, ω_nodes, ηR1, ηa1, ηL1,
        ηu1, ηB1, ηG1
    )
end

#%% cell
#=
Finally, we construct the `Model` type, which has an instance of `Params`,
`SteadyState` and `Grids` as its three fields.
=#

#%% cell
struct Model
    p::Params
    s::SteadyState
    g::Grids
end

function Model(;grid_source=:mat, kwargs...)
    p = Params(;kwargs...)
    s = SteadyState(p)
    g = Grids(p, s; grid_source=grid_source)
    Model(p, s, g)
end

Base.show(io::IO, m::Model) = println(io, "New Keynesian model")

#%% skip
for T in (:Params, :SteadyState, :Grids)
    @eval function Base.show(io::IO, x::$(T))
        dump(IOContext(io, :limit => true), x)
    end
end
