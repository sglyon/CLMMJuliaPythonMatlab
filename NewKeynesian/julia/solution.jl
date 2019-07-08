#%% cell
#=
Now that we have a model, we will construct one more helper function that takes
the control variables $(S, F, C)$ and shocks $(\delta, R, \eta_G, \eta_a,
\eta_L, \eta_R)$ and applies equilibrium conditions to produce $(\pi, \delta',
Y, L, Y_n, R')$. We will use this function in both the solution and simulation
routines below.
=#

#%% cell
function Base.step(m::Model, S, F, C, δ0, R0, ηG, ηa, ηL, ηR)
    @unpack Θ, ϵ, gbar, ϑ, γ, β, μ, ϕ_π, ϕ_y, πstar = m.p
    πs = m.s.π

    # Compute pie(t) from condition (35) in MM (2015)
    π0 = @. ((1-(1-Θ)*(S/F)^(1-ϵ))/Θ)^(1/(ϵ-1))

    # Compute delta(t) from condition (36) in MM (2015)
    δ1 = @. ((1-Θ)*((1-Θ*π0^(ϵ-1))/(1-Θ))^(ϵ/(ϵ-1))+Θ*π0^ϵ/δ0)^(-1.0)

    # Compute Y(t) from condition (38) in MM (2015)
    Y0 = @. C/(1-gbar/exp(ηG))

    # Compute L(t) from condition (37) in MM (2015)
    L0 = @. Y0/exp(ηa)/δ1

    #  Compute Yn(t) from condition (31) in MM (2015)
    Yn0 = @. (exp(ηa)^(1+ϑ)*(1-gbar/exp(ηG))^(-γ)/exp(ηL))^(1/(ϑ+γ))

    # Compute R(t) from conditions (27), (39) in MM (2015) -- Taylor rule
    R1 = @. πstar / β * (R0*β/πstar)^μ*((π0/πstar)^ϕ_π * (Y0/Yn0)^ϕ_y)^(1-μ)*exp(ηR)

    π0, δ1, Y0, L0, Yn0, R1
end
#%% cell
#=
### Solution routine
=#

#%% cell
# construct an initial guess for the solution
function initial_coefs(m::Model, degree)
    npol = size(m.g.X0_G[degree], 2)
    coefs = fill(1e-5, npol, 3)
    coefs[1, :] = [m.s.S, m.s.F, m.s.C^(-m.p.γ)]
    coefs
end

function solve(m::Model; verbose::Bool=false)
    # simplify notation
    n, n_nodes = size(m.g.ηR, 1), length(m.g.ω_nodes)

    ## allocate memory
    # euler equations
    e = zeros(n, 3)

    # previous iteration S, F, C
    S0_old_G = ones(n)
    F0_old_G = ones(n)
    C0_old_G = ones(n)

    # current iteration S, F, C
    S0_new_G = ones(n)
    F0_new_G = ones(n)
    C0_new_G = ones(n)

    # future S, F, C
    S1 = zeros(n, n_nodes)
    F1 = zeros(n, n_nodes)
    C1 = zeros(n, n_nodes)
    π1 = zeros(n, n_nodes)

    local coefs, start_time

    degs = m.p.deg > 1 ? [1, m.p.deg] : [m.p.deg]

    for deg in degs
        # set up matrices for this degree
        X0_G = m.g.X0_G[deg]

        # future basis matrix for S, F, C
        X1 = Array{Float64}(undef, n, n_complete(8, deg))

        # initialize coefs
        if deg <= 1
            coefs = initial_coefs(m, deg)
        else
            # for higher order, start with degree 2 coefficients
            coefs = X0_G\e
            # old_coefs = copy(coefs)
            # coefs = initial_coefs(m, deg)
            # coefs[1:size(old_coefs, 1), :] = old_coefs
        end

        err = 1.0
        it = 0

        start_time = time()
        # solve at this degree of complete polynomial
        while err > m.p.tol
            it += 1;
            # Current choices (at t)
            # ------------------------------
            S0 = X0_G*coefs[:, 1]                # Compute S(t) using coefs
            F0 = X0_G*coefs[:, 2]                # Compute F(t) using coefs
            C0 = X0_G*coefs[:, 3]
            C0 .^= (-1/m.p.γ)                    # Compute C(t) using coefs

            π0, δ1, Y0, L0, Yn0, R1 = step(m, S0, F0, C0, m.g.δ, m.g.R, m.g.ηG,
                                           m.g.ηa, m.g.ηL, m.g.ηR)

            if m.p.zlb R1 .= max.(R1, 1.0) end

            for u in 1:n_nodes
                # Form complete polynomial of degree "Degree" (at t+1) on future state
                complete_polynomial!(
                    X1,
                    hcat(log.(R1), log.(δ1), m.g.ηR1[:, u], m.g.ηa1[:, u],
                         m.g.ηL1[:, u], m.g.ηu1[:, u], m.g.ηB1[:, u], m.g.ηG1[:, u]),
                    deg
                )

                S1[:, u] = X1*coefs[:, 1]                # Compute S(t+1)
                F1[:, u] = X1*coefs[:, 2]                # Compute F(t+1)
                C1[:, u] = (X1*coefs[:, 3]).^(-1/m.p.γ)  # Compute C(t+1)
            end

            # Compute next-period π using condition
            # (35) in MM (2015)
            @. π1 = ((1-(1-m.p.Θ).*(S1./F1).^(1-m.p.ϵ))/m.p.Θ).^(1/(m.p.ϵ-1))

            # Evaluate conditional expectations in the Euler equations
            #---------------------------------------------------------
            e[:, 1] = @.(exp(m.g.ηu)*exp(m.g.ηL)*L0^m.p.ϑ*Y0/exp(m.g.ηa) + (m.p.β*m.p.Θ*π1^m.p.ϵ*S1))*m.g.ω_nodes
            e[:, 2] = @.(exp(m.g.ηu)*C0^(-m.p.γ)*Y0 + (m.p.β*m.p.Θ*π1^(m.p.ϵ-1)*F1))*m.g.ω_nodes
            e[:, 3] = @.(m.p.β*exp(m.g.ηB)/exp(m.g.ηu)*R1*((exp(m.g.ηu1).*C1^(-m.p.γ)/π1)))*m.g.ω_nodes

            # Variables of the current iteration
            #-----------------------------------
            copy!(S0_new_G, S0)
            copy!(F0_new_G, F0)
            copy!(C0_new_G, C0)

            # Compute and update the coefficients of the decision functions
            # -------------------------------------------------------------
            coefs_hat = X0_G\e   # Compute the new coefficients of the decision
                                 # functions using a backslash operator

            # Update the coefficients using damping
            coefs = m.p.damp*coefs_hat + (1-m.p.damp)*coefs

            # Evaluate the percentage (unit-free) difference between the values
            # on the grid from the previous and current iterations
            # -----------------------------------------------------------------
            # The convergence criterion is adjusted to the damping parameters
            err = mean(abs, 1.0.-S0_new_G./S0_old_G) +
                  mean(abs, 1.0.-F0_new_G./F0_old_G) +
                  mean(abs, 1.0.-C0_new_G./C0_old_G)

            if (it % 20 == 0) && verbose
                @printf "On iteration %d err is %6.7e\n" it err
            end

            # Store the obtained values for S(t), F(t), C(t) on the grid to
            # be used on the subsequent iteration in Section 10.2.6
            #-----------------------------------------------------------------------
            copy!(S0_old_G, S0_new_G)
            copy!(F0_old_G, F0_new_G)
            copy!(C0_old_G, C0_new_G)

        end
    end

    coefs, time() - start_time
end

#%% cell
#=
### Simulation
=#

#%% cell
struct Simulation
    # shocks
    ηR::Vector{Float64}
    ηa::Vector{Float64}
    ηL::Vector{Float64}
    ηu::Vector{Float64}
    ηB::Vector{Float64}
    ηG::Vector{Float64}

    # variables
    δ::Vector{Float64}
    R::Vector{Float64}
    S::Vector{Float64}
    F::Vector{Float64}
    C::Vector{Float64}
    π::Vector{Float64}
    Y::Vector{Float64}
    L::Vector{Float64}
    Yn::Vector{Float64}
    w::Vector{Float64}
end

function Simulation(m::Model, coefs::Matrix)

    # 11. Simualating a time-series solution
    #---------------------------------------
    _path = joinpath(@__DIR__, "..", "epsi_test_NK.mat")
    rands = (matread(_path)["epsi_test_NK"])::Matrix{Float64}
    capT = size(rands, 1)

    # Initialize the values of 6 exogenous shocks
    #--------------------------------------------
    ηR = zeros(capT)
    ηa = zeros(capT)
    ηL = zeros(capT)
    ηu = zeros(capT)
    ηB = zeros(capT)
    ηG = zeros(capT)

    # Generate the series for shocks
    #-------------------------------
    @inbounds for t in 1:capT-1
        ηR[t+1] = m.p.ρηR*ηR[t] + m.p.σηR*rands[t, 1]
        ηa[t+1] = m.p.ρηa*ηa[t] + m.p.σηa*rands[t, 2]
        ηL[t+1] = m.p.ρηL*ηL[t] + m.p.σηL*rands[t, 3]
        ηu[t+1] = m.p.ρηu*ηu[t] + m.p.σηu*rands[t, 4]
        ηB[t+1] = m.p.ρηB*ηB[t] + m.p.σηB*rands[t, 5]
        ηG[t+1] = m.p.ρηG*ηG[t] + m.p.σηG*rands[t, 6]
    end

    δ  = ones(capT+1) # Time series of delta(t)
    R  = ones(capT+1) # Time series of R(t)
    S  = Array{Float64}(undef, capT)   # Time series of S(t)
    F  = Array{Float64}(undef, capT)   # Time series of F(t)
    C  = Array{Float64}(undef, capT)   # Time series of C(t)
    π  = Array{Float64}(undef, capT)   # Time series of π(t)
    Y  = Array{Float64}(undef, capT)   # Time series of Y(t)
    L  = Array{Float64}(undef, capT)   # Time series of L(t)
    Yn = Array{Float64}(undef, capT)   # Time series of Yn(t)
    w  = Array{Float64}(undef, capT)

    pol_bases = Array{Float64}(undef, 1, size(coefs, 1))
    @inbounds for t in 1:capT
        # Construct the matrix of explanatory variables "pol_bases" on the series
        # of state variables; columns of "pol_bases" are given by the basis
        # functions of the polynomial of degree 2
        complete_polynomial!(
            pol_bases,
            hcat(log(R[t]), log(δ[t]), ηR[t], ηa[t], ηL[t], ηu[t], ηB[t], ηG[t]),
            m.p.deg
        )
        S[t], F[t], MU = pol_bases*coefs
        C[t] = (MU).^(-1/m.p.γ)

        π[t], δ[t+1], Y[t], L[t], Yn[t], R[t+1] = step(m, S[t], F[t], C[t], δ[t],
                                                       R[t], ηG[t], ηa[t], ηL[t], ηR[t])

        # Compute real wage
        w[t] = exp(ηL[t])*(L[t]^m.p.ϑ)*(C[t]^m.p.γ)

        # If ZLB is imposed, set R(t)=1 if ZLB binds
        if m.p.zlb; R[t+1] = max(R[t+1],1.0); end
    end

    Simulation(ηR, ηa, ηL, ηu, ηB, ηG, δ, R, S, F, C, π, Y, L, Yn, w)
end

#%% cell
#=
### Accuracy
=#

#%% cell
struct Residuals
    resids::Matrix{Float64}
end

function Residuals(m::Model, coefs::Matrix, sim::Simulation; burn::Int=200)
    capT = length(sim.w)
    resids = zeros(9, capT)

    # Integration method for evaluating accuracy
    # ------------------------------------------
    # Monomial integration rule with 2N^2+1 nodes
    ϵ_nodes, ω_nodes = qnwmonomial2(vcov(m.p))
    n_nodes = length(ω_nodes)

    # Allocate for arrays needed in the loop
    basis_mat = Array{Float64}(undef, n_nodes, 8)
    X1 = Array{Float64}(undef, n_nodes, size(coefs, 1))

    ηR1 = Array{Float64}(undef, n_nodes)
    ηa1 = Array{Float64}(undef, n_nodes)
    ηL1 = Array{Float64}(undef, n_nodes)
    ηu1 = Array{Float64}(undef, n_nodes)
    ηB1 = Array{Float64}(undef, n_nodes)
    ηG1 = Array{Float64}(undef, n_nodes)

    for t in 1:capT                 # For each given point,
        # Take the corresponding value for shocks at t
        #---------------------------------------------
        ηR0 = sim.ηR[t]  # ηR(t)
        ηa0 = sim.ηa[t]  # ηa(t)
        ηL0 = sim.ηL[t]  # ηL(t)
        ηu0 = sim.ηu[t]  # ηu(t)
        ηB0 = sim.ηB[t]  # ηB(t)
        ηG0 = sim.ηG[t]  # ηG(t)

        # Exctract time t values for all other variables (and t+1 for R, δ)
        #------------------------------------------------------------------
        R0  = sim.R[t]   # R(t-1)
        δ0  = sim.δ[t]   # δ(t-1)
        R1  = sim.R[t+1] # R(t)
        δ1  = sim.δ[t+1] # δ(t)

        L0  = sim.L[t]   # L(t)
        Y0  = sim.Y[t]   # Y(t)
        Yn0 = sim.Yn[t]  # Yn(t)
        π0  = sim.π[t]   # π(t)
        S0  = sim.S[t]   # S(t)
        F0  = sim.F[t]   # F(t)
        C0  = sim.C[t]   # C(t)

        # Fill basis matrix with R1, δ1 and shocks
        #-----------------------------------------
        # Note that we do not premultiply by standard deviations as ϵ_nodes
        # already include them. All these variables are vectors of length n_nodes
        copy!(ηR1, ηR0*m.p.ρηR .+ ϵ_nodes[:, 1])
        copy!(ηa1, ηa0*m.p.ρηa .+ ϵ_nodes[:, 2])
        copy!(ηL1, ηL0*m.p.ρηL .+ ϵ_nodes[:, 3])
        copy!(ηu1, ηu0*m.p.ρηu .+ ϵ_nodes[:, 4])
        copy!(ηB1, ηB0*m.p.ρηB .+ ϵ_nodes[:, 5])
        copy!(ηG1, ηG0*m.p.ρηG .+ ϵ_nodes[:, 6])

        basis_mat[:, 1] .= log.(R1)
        basis_mat[:, 2] .= log.(δ1)
        basis_mat[:, 3] = ηR1
        basis_mat[:, 4] = ηa1
        basis_mat[:, 5] = ηL1
        basis_mat[:, 6] = ηu1
        basis_mat[:, 7] = ηB1
        basis_mat[:, 8] = ηG1

        # Future choices at t+1
        #----------------------
        # Form a complete polynomial of degree "Degree" (at t+1) on future state
        # variables; n_nodes-by-npol
        complete_polynomial!(X1, basis_mat, m.p.deg)

        # Compute S(t+1), F(t+1) and C(t+1) in all nodes using coefs
        S1 = X1*coefs[:, 1]
        F1 = X1*coefs[:, 2]
        C1 = (X1*coefs[:, 3]).^(-1/m.p.γ)

        # Compute π(t+1) using condition (35) in MM (2015)
        π1 = @. ((1-(1-m.p.Θ)*(S1./F1).^(1-m.p.ϵ))/m.p.Θ).^(1/(m.p.ϵ-1))

        # Compute residuals for each of the 9 equilibrium conditions
        #-----------------------------------------------------------
        resids[1, t] = 1-dot(ω_nodes,
            @. (exp(ηu0)*exp(ηL0)*L0^m.p.ϑ*Y0/exp(ηa0) + m.p.β*m.p.Θ*π1.^m.p.ϵ.*S1)/S0
        )
        resids[2, t] = 1 - dot(ω_nodes,
            @. (exp(ηu0)*C0^(-m.p.γ)*Y0 + m.p.β*m.p.Θ*π1.^(m.p.ϵ-1).*F1)/F0
        )
        resids[3, t] = 1.0 -dot(ω_nodes,
            @. (m.p.β*exp(ηB0)/exp(ηu0)*R1*exp.(ηu1).*C1.^(-m.p.γ)./π1)/C0^(-m.p.γ)
        )
        resids[4, t] = 1-((1-m.p.Θ*π0^(m.p.ϵ-1))/(1-m.p.Θ))^(1/(1-m.p.ϵ))*F0/S0
        resids[5, t] = 1-((1-m.p.Θ)*((1-m.p.Θ*π0^(m.p.ϵ-1))/(1-m.p.Θ))^(m.p.ϵ/(m.p.ϵ-1)) + m.p.Θ*π0^m.p.ϵ/δ0)^(-1)/δ1
        resids[6, t] = 1-exp(ηa0)*L0*δ1/Y0
        resids[7, t] = 1-(1-m.p.gbar/exp(ηG0))*Y0/C0
        resids[8, t] = 1-(exp(ηa0)^(1+m.p.ϑ)*(1-m.p.gbar/exp(ηG0))^(-m.p.γ)/exp(ηL0))^(1/(m.p.ϑ+m.p.γ))/Yn0
        resids[9, t] = 1-m.s.π/m.p.β*(R0*m.p.β/m.s.π)^m.p.μ*((π0/m.s.π)^m.p.ϕ_π * (Y0/Yn0)^m.p.ϕ_y)^(1-m.p.μ)*exp(ηR0)/R1   # Taylor rule

        # If the ZLB is imposed and R>1, the residuals in the Taylor rule (the
        # 9th equation) are zero
        if m.p.zlb && R1 <= 1; resids[9, t] = 0.0; end

    end
    # discard the first burn observations
    Residuals(resids[:, burn+1:end])
end

Statistics.mean(r::Residuals) = log10(mean(abs, r.resids))
Base.max(r::Residuals) = log10(maximum(abs, r.resids))
max_E(r::Residuals) = log10.(maximum(abs, r.resids, dims=2))[:]
