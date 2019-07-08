#
# Various update methods depending on the solution method
#

# For iterating on initial guess and used within PFI
function update!(V::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{IterateOnPolicy},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Get sizes and allocate for complete_polynomial
    kgrid = ncgm.kgrid; zgrid = ncgm.zgrid;
    nk, nz = length(kgrid), length(zgrid)

    _inds = LinearIndices((nk, nz))

    # Iterate over all states
    for ik in 1:nk, iz in 1:nz
        # Pull out states
        k = kgrid[ik]
        z = zgrid[iz]

        # Pull out policy and evaluate consumption
        ikiz_index = _inds[ik, iz]
        k1 = kpol[ikiz_index]
        c = expendables_t(ncgm, k, z) - k1

        # New value
        EV = compute_EV(ncgm, vp, k1, iz)
        V[ikiz_index] = u(ncgm, c) + ncgm.β*EV
    end

    # Update coefficients
    update_v!(vp, Φ \ V, 1.0)
    update_k!(vp, Φ \ kpol, 1.0)

    return V
end

# ECM_VFI
function update!(V::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{VFI_ECM},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Get sizes and allocate for complete_polynomial
    kgrid = ncgm.kgrid; zgrid = ncgm.zgrid;
    nk, nz = length(kgrid), length(zgrid)

    # Iterate over all states
    temp = Array{Float64}(undef, n_complete(2, vp.d))
    _inds = LinearIndices((nk, nz))
    for ik in 1:nk, iz in 1:nz
        ikiz_index = _inds[ik, iz]
        k = kgrid[ik]
        z = zgrid[iz]

        # Policy from envelope condition
        kp = env_condition_kp!(temp, ncgm, vp, k, z)
        c = expendables_t(ncgm, k, z) - kp
        kpol[ikiz_index] = kp

        # New value
        EV = compute_EV!(temp, ncgm, vp, kp, iz)
        V[ikiz_index] = u(ncgm, c) + ncgm.β*EV
    end

    # Update coefficients
    update_v!(vp, Φ \ V, 1.0)
    update_k!(vp, Φ \ kpol, 1.0)

    return V
end

# VFI_EGM
function update!(V::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{VFI_EGM},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Get sizes and allocate for complete_polynomial
    kgrid = ncgm.kgrid; zgrid = ncgm.zgrid; grid = ncgm.grid;
    nk, nz = length(kgrid), length(zgrid)

    # Iterate
    temp = Array{Float64}(undef, n_complete(2, vp.d))
    _inds = LinearIndices((nk, nz))
    for iz=1:nz, ik=1:nk

        # In EGM we use the grid points as if they were our
        # policy for yesterday and find implied kt
        ikiz_index = _inds[ik, iz]
        k1 = kgrid[ik];z = zgrid[iz];

        # Compute the derivative of expected values
        dEV = compute_dEV!(temp, ncgm, vp, k1, iz)

        # Compute optimal consumption
        c = duinv(ncgm, ncgm.β*dEV)

        # Need to find corresponding kt for optimal c
        obj(kt) = expendables_t(ncgm, kt, z) - c - k1
        kt_star = brent(obj, 0.0, 2.0, xtol=1e-10)

        # New value
        EV = compute_EV!(temp, ncgm, vp, k1, iz)
        V[ikiz_index] = u(ncgm, c) + ncgm.β*EV
        kpol[ikiz_index] = kt_star
    end

    # New Φ (has our new "kt_star" and z points)
    Φ_egm = complete_polynomial([kpol grid[:, 2]], vp.d)

    # Update coefficients
    update_v!(vp, Φ_egm \ V, 1.0)
    update_k!(vp, Φ_egm \ grid[:, 1], 1.0)

    # Update V and kpol to be value and policy corresponding
    # to our grid again
    copy!(V, Φ*vp.v_coeffs)
    copy!(kpol, Φ*vp.k_coeffs)

    return V
end

# VFI
function update!(V::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{VFI},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Get sizes and allocate for complete_polynomial
    kgrid = ncgm.kgrid; zgrid = ncgm.zgrid
    nk, nz = length(kgrid), length(zgrid)

    # Iterate over all states
    temp = Array{Float64}(undef, n_complete(2, vp.d))
    _inds = LinearIndices((nk, nz))
    for iz=1:nz, ik=1:nk
        k = kgrid[ik]; z = zgrid[iz]

        # Define an objective function (negative for minimization)
        y = expendables_t(ncgm, k, z)
        solme(kp) = du(ncgm, y - kp) - ncgm.β*compute_dEV!(temp, ncgm, vp, kp, iz)

        # Find sol to foc
        kp = brent(solme, 1e-8, y-1e-8; rtol=1e-12)
        c = expendables_t(ncgm, k, z) - kp

        # New value
        ikiz_index = _inds[ik, iz]
        EV = compute_EV!(temp, ncgm, vp, kp, iz)
        V[ikiz_index] = u(ncgm, c) + ncgm.β*EV
        kpol[ikiz_index] = kp
    end

    # Update coefficients
    update_v!(vp, Φ \ V, 1.0)
    update_k!(vp, Φ \ kpol, 1.0)

    return V
end

# PFI ECM
function update!(V::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{PFI_ECM},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Copy valuecoeffs object and use to iterate to
    # convergence given a policy
    vp_igp = copy(vp, IterateOnPolicy())
    solve(ncgm, vp_igp; nskipprint=1000, maxiter=5000, verbose=false)

    # Update the policy and values
    kp = env_condition_kp(ncgm, vp)
    update_k!(vp, Φ \ kp, 1.0)
    update_v!(vp, vp_igp.v_coeffs, 1.0)

    # Update all elements of value
    copy!(V, Φ*vp.v_coeffs)
    copy!(kpol, kp)

    return V
end

# Conventional PFI
function update!(V::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{PFI},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Get sizes and allocate for complete_polynomial
    kgrid = ncgm.kgrid; zgrid = ncgm.zgrid; grid = ncgm.grid;
    nk, nz = length(kgrid), length(zgrid)

    # Copy valuecoeffs object and use to iterate to
    # convergence given a policy
    vp_igp = copy(vp, IterateOnPolicy())
    solve(ncgm, vp_igp; nskipprint=1000, maxiter=5000, verbose=false)

    # Update the policy and values
    temp = Array{Float64}(undef, n_complete(2, vp.d))
    _inds = LinearIndices((nk, nz))
    for ik in 1:nk, iz in 1:nz
        k = kgrid[ik]; z = zgrid[iz];

        # Define an objective function (negative for minimization)
        y = expendables_t(ncgm, k, z)
        solme(kp) = du(ncgm, y - kp) - ncgm.β*compute_dEV!(temp, ncgm, vp, kp, iz)

        # Find minimum of objective
        kp = brent(solme, 1e-8, y-1e-8; rtol=1e-12)

        # Update policy function
        ikiz_index = _inds[ik, iz]
        kpol[ikiz_index] = kp
    end

    # Get new coeffs
    update_k!(vp, Φ \ kpol, 1.0)
    update_v!(vp, vp_igp.v_coeffs, 1.0)

    # Update all elements of value
    copy!(V, Φ*vp.v_coeffs)

    return V
end

# Envelope condition iterate on derivative
function update!(dV::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{dVFI_ECM},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Get sizes and allocate for complete_polynomial
    kgrid = ncgm.kgrid; zgrid = ncgm.zgrid; grid = ncgm.grid;
    nk, nz, ns = length(kgrid), length(zgrid), size(grid, 1)

    # Iterate over all states
    temp = Array{Float64}(undef, n_complete(2, vp.d))
    _inds = LinearIndices((nk, nz))
    for iz=1:nz, ik=1:nk
        k = kgrid[ik]; z = zgrid[iz];

        # Envelope condition implies optimal kp
        kp = env_condition_kp!(temp, ncgm, vp, k, z)
        c = expendables_t(ncgm, k, z) - kp

        # New value
        ikiz_index = _inds[ik, iz]
        dEV = compute_dEV!(temp, ncgm, vp, kp, iz)
        dV[ikiz_index] = (1-ncgm.δ+df(ncgm, k, z))*ncgm.β*dEV
        kpol[ikiz_index] = kp
    end

    # Get new coeffs
    update_k!(vp, Φ \ kpol, 1.0)
    update_v!(vp, dΦ \ dV, 1.0)

    return dV
end

# Conventional Euler equation method
function update!(V::Vector{Float64}, kpol::Vector{Float64},
                 ncgm::NeoclassicalGrowth, vp::ValueCoeffs{EulEq},
                 Φ::Matrix{Float64}, dΦ::Matrix{Float64})
    # Get sizes and allocate for complete_polynomial
    @unpack kgrid, zgrid, weights, z1 = ncgm
    nz1, nz = size(z1)
    nk = length(kgrid)

    # Iterate over all states
    temp = Array{Float64}(undef, n_complete(2, vp.d))
    _inds = LinearIndices((nk, nz))
    for iz in 1:nz, ik in 1:nk
        k = kgrid[ik]; z = zgrid[iz];

        # Create current polynomial
        complete_polynomial!(temp, [k, z], vp.d)

        # Compute what capital will be tomorrow according to policy
        kp = dot(temp, vp.k_coeffs)

        # Compute RHS of EE
        rhs_ee = 0.0
        for iz1 in 1:nz1
            # Possible z in t+1
            zp = z1[iz1, iz]

            # Policy for k_{t+2}
            complete_polynomial!(temp, [kp, zp], vp.d)
            kpp = dot(temp, vp.k_coeffs)

            # Implied t+1 consumption
            cp = expendables_t(ncgm, kp, zp) - kpp

            # Add to running expectation
            rhs_ee += ncgm.β*weights[iz1]*du(ncgm, cp)*(1-ncgm.δ+df(ncgm, kp, zp))
        end

        # The rhs of EE implies consumption and investment in t
        c = duinv(ncgm, rhs_ee)
        kp_star = expendables_t(ncgm, k, z) - c

        # New value
        ikiz_index = _inds[ik, iz]
        EV = compute_EV!(temp, ncgm, vp, kp_star, iz)
        V[ikiz_index] = u(ncgm, c) + ncgm.β*EV
        kpol[ikiz_index] = kp_star
    end

    # Update coefficients
    update_v!(vp, Φ \ V, 1.0)
    update_k!(vp, Φ \ kpol, 1.0)

    return V
end

#
# General solve method
#

function solve(
        ncgm::NeoclassicalGrowth, vp::ValueCoeffs;
        tol::Float64=1e-6, maxiter::Int=5000, dampen::Float64=1.0,
        nskipprint::Int=1, verbose::Bool=true
    )
    # Get number of k and z on grid
    nk, nz = length(ncgm.kgrid), length(ncgm.zgrid)

    # Build basis matrix and value function
    dPhi = complete_polynomial(ncgm.grid, vp.d, Derivative{1}())
    Phi = complete_polynomial(ncgm.grid, vp.d)
    V = build_V_or_dV(ncgm, vp)
    k = build_k(ncgm, vp)
    Vnew = copy(V)
    knew = copy(k)

    # Print column names
    if verbose
        @printf("| Iteration | Distance V | Distance K |\n")
    end

    # Iterate to convergence
    dist, iter = 10.0, 0
    while (tol < dist) & (iter < maxiter)
        # Update the value function using appropriate update methods
        update!(Vnew, knew, ncgm, vp, Phi, dPhi)

        # Compute distance and update all relevant elements
        iter += 1
        dist_v = maximum(abs, 1.0 .- Vnew./V)
        dist_k = maximum(abs, 1.0 .- knew./k)
        copy!(V, Vnew)
        copy!(k, knew)

        # If we are iterating on a policy, use the difference of values
        # otherwise use the distance on policy
        dist = ifelse(solutionmethod(vp) == IterateOnPolicy, dist_v, dist_k)

        # Print status update
        if verbose && (iter%nskipprint == 0)
            @printf("|%-11d|%-12e|%-12e|\n", iter, dist_v, dist_k)
        end
    end

    # Update value and policy functions one last time as long as the
    # solution method isn't IterateOnPolicy
    if ~(solutionmethod(vp) == IterateOnPolicy)
        # Update capital policy after finished
        kp = env_condition_kp(ncgm, vp)
        update_k!(vp, complete_polynomial(ncgm.grid, vp.d) \ kp, 1.0)

        # Update value function according to specified policy
        vp_igp = copy(vp, IterateOnPolicy())
        solve(ncgm, vp_igp; tol=1e-10, maxiter=5000, verbose=false)
        update_v!(vp, vp_igp.v_coeffs, 1.0)

    end

    return vp
end
