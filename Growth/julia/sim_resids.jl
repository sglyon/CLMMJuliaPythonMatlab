"""
Simulates the neoclassical growth model for a given set of solution
coefficients. It simulates for `capT` periods and discards first
`nburn` observations.
"""
function simulate(ncgm::NeoclassicalGrowth, vp::ValueCoeffs,
                  shocks::Vector{Float64}; capT::Int=10_000,
                  nburn::Int=200)
    # Unpack parameters
    kp = 0.0  # Policy holder
    temp = Array{Float64}(undef, n_complete(2, vp.d))

    # Allocate space for k and z
    ksim = Array{Float64}(undef, capT+nburn)
    zsim = Array{Float64}(undef, capT+nburn)

    # Initialize both k and z at 1
    ksim[1] = 1.0
    zsim[1] = 1.0

    # Simulate
    temp = Array{Float64}(undef, n_complete(2, vp.d))
    for t in 2:capT+nburn
        # Evaluate k_t given yesterday's (k_{t-1}, z_{t-1})
        kp = env_condition_kp!(temp, ncgm, vp, ksim[t-1], zsim[t-1])

        # Draw new z and update k using policy above
        zsim[t] = zsim[t-1]^ncgm.ρ * exp(ncgm.σ*shocks[t])
        ksim[t] = kp
    end

    return ksim[nburn+1:end], zsim[nburn+1:end]
end

function simulate(ncgm::NeoclassicalGrowth, vp::ValueCoeffs;
                  capT::Int=10_000, nburn::Int=200, seed=42)
    Random.seed!(seed)  # Set specific seed
    shocks = randn(capT + nburn)

    return simulate(ncgm, vp, shocks; capT=capT, nburn=nburn)
end

"""
This function evaluates the Euler Equation residual for a single point (k, z)
"""
function EulerEquation!(out::Vector{Float64}, ncgm::NeoclassicalGrowth,
                        vp::ValueCoeffs, k::Float64, z::Float64,
                        nodes::Vector{Float64}, weights::Vector{Float64})
    # Evaluate consumption today
    k1 = env_condition_kp!(out, ncgm, vp, k, z)
    c = expendables_t(ncgm, k, z) - k1
    LHS = du(ncgm, c)

    # For each of realizations tomorrow, evaluate expectation on RHS
    RHS = 0.0
    for (eps, w) in zip(nodes, weights)
        # Compute ztp1
        z1 = z^ncgm.ρ * exp(eps)

        # Evaluate the ktp2
        ktp2 = env_condition_kp!(out, ncgm, vp, k1, z1)

        # Get c1
        c1 = expendables_t(ncgm, k1, z1) - ktp2

        # Update RHS of equation
        RHS = RHS + w*du(ncgm, c1)*(1 - ncgm.δ + df(ncgm, k1, z1))
    end

    return abs(ncgm.β*RHS/LHS - 1.0)
end

"""
Given simulations for k and z, it computes the euler equation residuals
along the entire simulation. It reports the mean and max values in
log10.
"""
function ee_residuals(ncgm::NeoclassicalGrowth, vp::ValueCoeffs,
                      ksim::Vector{Float64}, zsim::Vector{Float64}; Qn::Int=10)
    # Figure out how many periods we simulated for and make sure k and z
    # are same length
    capT = length(ksim)
    @assert length(zsim) == capT

    # Finer integration nodes
    eps_nodes, weight_nodes = qnwnorm(Qn, 0.0, ncgm.σ^2)
    temp = Array{Float64}(undef, n_complete(2, vp.d))

    # Compute EE for each period
    EE_resid = Array{Float64}(undef, capT)
    for t=1:capT
        # Pull out current state
        k, z = ksim[t], zsim[t]

        # Compute residual of Euler Equation
        EE_resid[t] = EulerEquation!(temp, ncgm, vp, k, z, eps_nodes, weight_nodes)
    end

    return EE_resid
end

function ee_residuals(ncgm::NeoclassicalGrowth, vp::ValueCoeffs; Qn::Int=10)
    # Simulate and then call other ee_residuals method
    ksim, zsim = simulate(ncgm, vp)

    return ee_residuals(ncgm, vp, ksim, zsim; Qn=Qn)
end
