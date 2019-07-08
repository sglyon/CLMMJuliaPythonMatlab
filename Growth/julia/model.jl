#=
Contains the pieces to build the stochastic Neoclassical growth model
described in the Arellano, Maliar, Maliar, Tsyrennikov 2016 paper.
=#

#
# The Model
#
"""
The stochastic Neoclassical growth model type contains parameters
which define the model

* α: Capital share in output
* β: Discount factor
* δ: Depreciation rate
* γ: Risk aversion
* ρ: Persistence of the log of the productivity level
* σ: Standard deviation of shocks to log productivity level
* A: Coefficient on C-D production function

* kgrid: Grid over capital
* zgrid: Grid over productivity
* grid: Grid of (k, z) pairs
* eps_nodes: Nodes used to integrate
* weights: Weights used to integrate
* z1: A grid of the possible z1s tomorrow given eps_nodes and zgrid
"""
@with_kw struct NeoclassicalGrowth
    # Parameters
    α::Float64 = 0.36
    β::Float64 = 0.99
    δ::Float64 = 0.02
    γ::Float64 = 2.0
    ρ::Float64 = 0.95
    σ::Float64 = 0.01
    A::Float64 = (1.0/β - (1 - δ)) / α

    # Grids
    kgrid::Vector{Float64} = collect(range(0.9, stop=1.1, length=10))
    zgrid::Vector{Float64} = collect(range(0.9, stop=1.1, length=10))
    grid::Matrix{Float64} = gridmake(kgrid, zgrid)
    eps_nodes::Vector{Float64} = qnwnorm(5, 0.0, σ^2)[1]
    weights::Vector{Float64} = qnwnorm(5, 0.0, σ^2)[2]
    z1::Matrix{Float64} = (zgrid.^(ρ))' .* exp.(eps_nodes)
end

# Helper functions
f(ncgm::NeoclassicalGrowth, k, z) = @. z * (ncgm.A * k^ncgm.α)
df(ncgm::NeoclassicalGrowth, k, z) = @. ncgm.α * z * (ncgm.A * k^(ncgm.α - 1.0))

u(ncgm::NeoclassicalGrowth, c) = c > 1e-10 ? @.(c^(1-ncgm.γ)-1)/(1-ncgm.γ) : -1e10
du(ncgm::NeoclassicalGrowth, c) = c > 1e-10 ? c.^(-ncgm.γ) : 1e10
duinv(ncgm::NeoclassicalGrowth, u) = u .^ (-1 / ncgm.γ)

expendables_t(ncgm::NeoclassicalGrowth, k, z) = (1-ncgm.δ)*k + f(ncgm, k, z)



# Types for solution methods
abstract type SolutionMethod end

struct IterateOnPolicy <: SolutionMethod end
struct VFI_ECM <: SolutionMethod end
struct VFI_EGM <: SolutionMethod end
struct VFI <: SolutionMethod end
struct PFI_ECM <: SolutionMethod end
struct PFI <: SolutionMethod end
struct dVFI_ECM <: SolutionMethod end
struct EulEq <: SolutionMethod end

#
# Type for Approximating Value and Policy
#
mutable struct ValueCoeffs{T <: SolutionMethod,D <: Degree}
    d::D
    v_coeffs::Vector{Float64}
    k_coeffs::Vector{Float64}
end

function ValueCoeffs(::Type{Val{d}}, method::T) where T <: SolutionMethod where d
    # Initialize two vectors of zeros
    deg = Degree{d}()
    n = n_complete(2, deg)
    v_coeffs = zeros(n)
    k_coeffs = zeros(n)

    return ValueCoeffs{T,Degree{d}}(deg, v_coeffs, k_coeffs)
end

function ValueCoeffs(
        ncgm::NeoclassicalGrowth, ::Type{Val{d}}, method::T
    ) where T <: SolutionMethod where d
    # Initialize with vector of zeros
    deg = Degree{d}()
    n = n_complete(2, deg)
    v_coeffs = zeros(n)

    # Policy guesses based on k and z
    k, z = ncgm.grid[:, 1], ncgm.grid[:, 2]
    css = ncgm.A - ncgm.δ
    yss = ncgm.A
    c_pol = f(ncgm, k, z) * (css/yss)

    # Figure out what kp is
    k_pol = expendables_t(ncgm, k, z) - c_pol
    k_coeffs = complete_polynomial(ncgm.grid, d) \ k_pol

    return ValueCoeffs{T,Degree{d}}(deg, v_coeffs, k_coeffs)
end

solutionmethod(::ValueCoeffs{T}) where T <:SolutionMethod = T

# A few copy methods to make life easier
Base.copy(vp::ValueCoeffs{T,D}) where T where D =
    ValueCoeffs{T,D}(vp.d, vp.v_coeffs, vp.k_coeffs)

function Base.copy(vp::ValueCoeffs{T1,D}, ::T2) where T1 where D where T2 <: SolutionMethod
    ValueCoeffs{T2,D}(vp.d, vp.v_coeffs, vp.k_coeffs)
end

function Base.copy(
        ncgm::NeoclassicalGrowth, vp::ValueCoeffs{T}, ::Type{Val{new_degree}}
    ) where T where new_degree
    # Build Value and policy matrix
    deg = Degree{new_degree}()
    V = build_V(ncgm, vp)
    k = build_k(ncgm, vp)

    # Build new Phi
    Phi = complete_polynomial(ncgm.grid, deg)
    v_coeffs = Phi \ V
    k_coeffs = Phi \ k

    return ValueCoeffs{T,Degree{new_degree}}(deg, v_coeffs, k_coeffs)
end

"""
Updates the coefficients for the value function inplace in `vp`
"""
function update_v!(vp::ValueCoeffs, new_coeffs::Vector{Float64}, dampen::Float64)
    vp.v_coeffs = (1-dampen)*vp.v_coeffs + dampen*new_coeffs
end

"""
Updates the coefficients for the policy function inplace in `vp`
"""
function update_k!(vp::ValueCoeffs, new_coeffs::Vector{Float64}, dampen::Float64)
    vp.k_coeffs = (1-dampen)*vp.k_coeffs + dampen*new_coeffs
end

"""
Builds either V or dV depending on the solution method that is given. If it
is a solution method that iterates on the derivative of the value function
then it will return derivative of the value function, otherwise the
value function itself
"""
build_V_or_dV(ncgm::NeoclassicalGrowth, vp::ValueCoeffs) =
    build_V_or_dV(ncgm, vp, solutionmethod(vp)())

build_V_or_dV(ncgm, vp::ValueCoeffs, ::SolutionMethod) = build_V(ncgm, vp)
build_V_or_dV(ncgm, vp::ValueCoeffs, T::dVFI_ECM) = build_dV(ncgm, vp)

function build_dV(ncgm::NeoclassicalGrowth, vp::ValueCoeffs)
    Φ = complete_polynomial(ncgm.grid, vp.d, Derivative{1}())
    Φ*vp.v_coeffs
end

function build_V(ncgm::NeoclassicalGrowth, vp::ValueCoeffs)
    Φ = complete_polynomial(ncgm.grid, vp.d)
    Φ*vp.v_coeffs
end

function build_k(ncgm::NeoclassicalGrowth, vp::ValueCoeffs)
    Φ = complete_polynomial(ncgm.grid, vp.d)
    Φ*vp.k_coeffs
end

#
# Helper functions
#
function compute_EV!(cp_kpzp::Vector{Float64}, ncgm::NeoclassicalGrowth,
                     vp::ValueCoeffs, kp, iz)
    # Pull out information from types
    z1, weightsz = ncgm.z1, ncgm.weights

    # Get number nodes
    nzp = length(weightsz)

    EV = 0.0
    for izp in 1:nzp
        zp = z1[izp, iz]
        complete_polynomial!(cp_kpzp, [kp, zp], vp.d)
        EV += weightsz[izp] * dot(vp.v_coeffs, cp_kpzp)
    end

    return EV
end

function compute_EV(ncgm::NeoclassicalGrowth, vp::ValueCoeffs, kp, iz)
    cp_kpzp = Array{Float64}(undef, n_complete(2, vp.d))

    return compute_EV!(cp_kpzp, ncgm, vp, kp, iz)
end

function compute_EV(ncgm::NeoclassicalGrowth, vp::ValueCoeffs)
    # Get length of k and z grids
    kgrid, zgrid = ncgm.kgrid, ncgm.zgrid
    nk, nz = length(kgrid), length(zgrid)
    temp = Array{Float64}(undef, n_complete(2, vp.d))

    # Allocate space to store EV
    EV = Array{Float64}(undef, nk*nz)

    _inds = LinearIndices((nk, nz))

    for ik in 1:nk, iz in 1:nz
        # Pull out states
        k = kgrid[ik]
        z = zgrid[iz]
        ikiz_index = _inds[ik, iz]

        # Pass to scalar EV
        complete_polynomial!(temp, [k, z], vp.d)
        kp = dot(vp.k_coeffs, temp)
        EV[ikiz_index] = compute_EV!(temp, ncgm, vp, kp, iz)
    end

    return EV
end


function compute_dEV!(cp_dkpzp::Vector,  ncgm::NeoclassicalGrowth,
                      vp::ValueCoeffs, kp, iz)
    # Pull out information from types
    z1, weightsz = ncgm.z1, ncgm.weights

    # Get number nodes
    nzp = length(weightsz)

    dEV = 0.0
    for izp in 1:nzp
        zp = z1[izp, iz]
        complete_polynomial!(cp_dkpzp, [kp, zp], vp.d, Derivative{1}())
        dEV += weightsz[izp] * dot(vp.v_coeffs, cp_dkpzp)
    end

    return dEV
end

function compute_dEV(ncgm::NeoclassicalGrowth, vp::ValueCoeffs, kp, iz)
    compute_dEV!(Array{Float64}(undef, n_complete(2, vp.d)), ncgm, vp, kp, iz)
end



function env_condition_kp!(cp_out::Vector{Float64}, ncgm::NeoclassicalGrowth,
                           vp::ValueCoeffs, k::Float64, z::Float64)
    # Compute derivative of VF
    dV = dot(vp.v_coeffs, complete_polynomial!(cp_out, [k, z], vp.d, Derivative{1}()))

    # Consumption is then computed as
    c = duinv(ncgm, dV / (1 - ncgm.δ .+ df(ncgm, k, z)))

    return expendables_t(ncgm, k, z) - c
end

function env_condition_kp(ncgm::NeoclassicalGrowth, vp::ValueCoeffs,
                          k::Float64, z::Float64)
    cp_out = Array{Float64}(undef, n_complete(2, vp.d))
    env_condition_kp!(cp_out, ncgm, vp, k, z)
end

function env_condition_kp(ncgm::NeoclassicalGrowth, vp::ValueCoeffs)
    # Pull out k and z from grid
    k = ncgm.grid[:, 1]
    z = ncgm.grid[:, 2]

    # Create basis matrix for entire grid
    dPhi = complete_polynomial(ncgm.grid, vp.d, Derivative{1}())

    # Compute consumption
    c = duinv(ncgm, (dPhi*vp.v_coeffs) ./ (1-ncgm.δ.+df(ncgm, k, z)))

    return expendables_t(ncgm, k, z) .- c
end
