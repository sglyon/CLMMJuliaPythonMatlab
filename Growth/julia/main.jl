#=
This file mimics the functionality from Main_6.m

It starts with a linear guess and builds up subsequently
higher degree approximations of the value function (or
its derivative). It uses the previous order's solution as
guess for the solution algorithm at the next highest order.
=#
include("NeoclassicalGrowth.jl")

using DelimitedFiles, Statistics  # sdlib
using BasisMatrices
using .ECMMethods

function main_for_method(sm::SolutionMethod, nd::Int=5, shocks=randn(capT+nburn);
              capT=10_000, nburn=200, tol=1e-9, maxiter=2500,
              nskipprint=25, verbose=true)
    # Create model
    ncgm = NeoclassicalGrowth()

    # Create initial quadratic guess
    vp = ValueCoeffs(ncgm, Val{2}, IterateOnPolicy())
    solve(ncgm, vp; tol=1e-6, verbose=false)

    # Allocate memory for timings
    times = Array{Float64}(undef, nd-1)
    sols = Array{ValueCoeffs}(undef, nd-1)
    mean_ees = Array{Float64}(undef, nd-1)
    max_ees = Array{Float64}(undef, nd-1)

    # Solve using the solution method for degree 2 to 5
    vp = copy(vp, sm)
    for d in 2:nd
        # Change degree of solution method
        vp = copy(ncgm, vp, Val{d})

        # Time the current method
        start_time = time()
        solve(ncgm, vp; tol=tol, maxiter=maxiter, nskipprint=nskipprint,
              verbose=verbose)
        end_time = time()

        # Save the time and solution
        times[d-1] = end_time - start_time
        sols[d-1] = vp

        # Simulate and compute EE
        ks, zs = simulate(ncgm, vp, shocks; capT=capT, nburn=nburn)
        resids = ee_residuals(ncgm, vp, ks, zs; Qn=10)
        mean_ees[d-1] = log10.(mean(abs.(resids)))
        max_ees[d-1] = log10.(maximum(abs, resids))
    end

    return sols, times, mean_ees, max_ees
end

function main()
    open("METHODSLOG.log", "w") do file

        shocks = vec(readdlm("../EE_SHOCKS.csv"))
        # for sol_method in [VFI(), VFI_ECM(), VFI_EGM()]
        # for sol_method in [PFI(), PFI_ECM(), dVFI_ECM(), EulEq()]
        for sol_method in [VFI_ECM(), VFI(), VFI_EGM(),
                        PFI(), PFI_ECM(), EulEq(), dVFI_ECM()]
            # Make sure everything is compiled
            main_for_method(sol_method, 5, shocks; maxiter=2, verbose=false)

            # Run for real
            s_sm, t_sm, mean_eem, max_eem = main_for_method(
                sol_method, 5, shocks; tol=1e-9, verbose=true
            )

            println(file, "Solution Method: $sol_method")
            for (d, t) in zip([2, 3, 4, 5], t_sm)
                println(file, "\tDegree $d took time $t")
                println(file, "\tMean & Max EE are: " *
                            "$(round(mean_eem[d-1], digits=3)) & $(round(max_eem[d-1], digits=3))")
            end
        end

    end  # open file
end
