from NeoclassicalGrowth import *

shocks = np.loadtxt("../EE_SHOCKS.csv", delimiter=",")

# Define model
ncgm = NeoclassicalGrowth()

# First guess
vp = IterateOnPolicy(ncgm, 2)
vp.solve(tol=1e-9)

# Run simulate and ee to get all relevant functions compiled
ks, zs = vp.simulate(100, 10, shocks=shocks)
mean_ee, max_ee = vp.ee_residuals(ks, zs, 10)


# File to save results
with open("METHODSLOG.log", "w") as _file:

    for sol_method in [VFI, VFI_ECM, VFI_EGM, PFI, PFI_ECM, dVFI_ECM, EulEq]:
    # for sol_method in [VFI, VFI_ECM, VFI_EGM]:
    # for sol_method in [PFI, PFI_ECM, dVFI_ECM, EulEq]:
        # Set prev sol as iterate on policy
        new_sol = vp
        _file.write("Solution Method: {}\n".format(sol_method))
        for d in range(2, 6):
            new_sol = sol_method(ncgm, d, new_sol)
            ts = time.time()
            new_sol.solve(tol=1e-9, verbose=True, nskipprint=25)
            time_took = time.time() - ts
            _file.write("\tDegree {} took {}\n".format(d, time_took))

            # Compute Euler Errors
            ks, zs = new_sol.simulate(10000, 200, shocks=shocks)
            mean_ee, max_ee = new_sol.ee_residuals(ks, zs, Qn=10)
            _file.write("\tMean and Max EE are {} & {}\n".format(mean_ee, max_ee))
            new_sol = new_sol

