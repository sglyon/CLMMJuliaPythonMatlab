module ECMMethods

# stdlib stuff
using Random, LinearAlgebra, Printf

# Packages we need
using BasisMatrices, Optim, QuantEcon, Parameters
using BasisMatrices: Degree, Derivative


# Export functions that need to be exposed
export NeoclassicalGrowth, ValueCoeffs, f, df, u, du, duinv, expendables_t,
       SolutionMethod, IterateOnPolicy, VFI_ECM, VFI_EGM, VFI, PFI_ECM, PFI,
       dVFI_ECM, EulEq, build_V, build_k, solve, copy, simulate,
       EulerEquation!, ee_residuals, compute_EV, env_condition_kp, update_v!

# Include our other files
include("model.jl")
include("solutionmethods.jl")
include("sim_resids.jl")

#=


shocks = vec(readcsv("../EE_SHOCKS.csv"))
ncgm = NeoclassicalGrowth()
@time begin
vp = ValueCoeffs(ncgm, Val{2}, IterateOnPolicy())
solve(ncgm, vp, 1e-7, 5000, 1.0, 200, false)
vp_vfi = copy(vp, VFI_ECM())
Profile.clear()
solve(ncgm, vp_vfi, 1e-7, 5000, 1.0, 50, false)
end  # time

=#

end  # Module
