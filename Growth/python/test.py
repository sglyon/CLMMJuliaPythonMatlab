import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import time
import quantecon as qe

from collections import namedtuple
from interpolation.complete_poly import (CompletePolynomial,
                                         n_complete, complete_polynomial,
                                         complete_polynomial_der,
                                         _complete_poly_impl,
                                         _complete_poly_impl_vec,
                                         _complete_poly_der_impl,
                                         _complete_poly_der_impl_vec)
from numba import jit, vectorize

from NeoclassicalGrowth import *


# Test all jitted helper functions
def test_utility():
    # Make sure evaluates to right numbers
    c = np.random.rand(5)
    for gamma in [2.0, 5.0, 10.0]:
        # Check to make sure u(1) = 0.0 for all gamma
        assert(np.abs(u(1.0, gamma) - 0.0) < 1e-10)

        # Check formula for random consumption and make sure it operates on vectors
        assert(np.max(np.abs(u(c, gamma) - (c**(1-gamma) - 1)/(1-gamma))) < 1e-10)

        # Check to make sure du is 1 when c=1
        assert(np.abs(du(1.0, gamma) - 1.0) < 1e-10)

        # Check formula and that derivative of utility can operate on vectors
        assert(np.max(np.abs(du(c, gamma) - c**(-gamma))) < 1e-10)

        # Check to make sure u^{-1} is 1 when u is 1
        assert(np.abs(duinv(1.0, gamma) - 1.0) < 1e-10)

        # Check formula and that inverse of derivative of utility
        # can operate on vectors
        assert(np.max(np.abs(duinv(du(c, gamma), gamma) - c)) < 1e-10)


def test_production():
    # Create vectors of k and z
    k = 1 + np.random.rand(5)
    z = 1 + np.random.rand(5)

    A, delta = 1.0, 0.025
    for alpha in [0.2, 0.35, 0.5]:
        # Check to make sure f(1, 1) = 1.0 for all gamma
        assert(np.abs(f(1.0, 1.0, A, alpha) - 1.0) < 1e-10)

        # Check formula for random (k, z) and make sure it operates on vectors
        assert(np.max(np.abs(f(k, z, A, alpha) - A*z*k**alpha)) < 1e-10)

        # Check to make sure df(1, 1) is alpha*A
        assert(np.abs(df(1.0, 1.0, A, alpha) - A*alpha) < 1e-10)

        # Check formula and that derivative of production can operate on vectors
        assert(np.max(np.abs(df(k, z, A, alpha) - A*alpha*z*k**(alpha-1))) < 1e-10)

        # Check to make sure expendables of (1, 1) are (1-delta) + 1
        assert(np.abs(expendables_t(1.0, 1.0, A, alpha, delta)-(2-delta)) < 1e-10)

        # Check formula and that expendables can operate on vectors
        assert(np.max(np.abs(expendables_t(k, z, A, alpha, delta) -
                             (1-delta)*k - f(k, z, A, alpha))) < 1e-10)


def test_simulate():
    # Create a params type
    params = Params(1.0, 0.33, 0.99, 0.025, 2.0, 0.9, 0.01)

    for deg in [1, 2, 4]:
        # Force the policy rule to be k_{t+1} = 0.75 k_t + 0.25 z_t
        k_coeffs = np.zeros(n_complete(2, deg))
        k_coeffs[1] = 0.75
        k_coeffs[2] = 0.25

        # First sequence of shocks is all zeros -- Both k and z should stay at 1
        T1 = 5
        e1 = np.zeros(T1)
        k1, z1 = jit_simulate_ncgm(params, deg, k_coeffs, T1, 0, e1)
        assert(np.max(np.abs(k1 - 1)) < 1e-10)
        assert(np.max(np.abs(z1 - 1)) < 1e-10)

        # Second sequence of shocks has a 1 std dev and then all zeros
        T2 = 10
        e2 = np.zeros(T2)
        e2[1] = 1.0
        k2, z2 = jit_simulate_ncgm(params, deg, k_coeffs, T2, 0, e2)
        assert(abs(z2[0] - 1.0) < 1e-10)
        assert(max([abs(z2[i+1] - np.exp(params.sigma)**(params.rho**i))
                    for i in range(T2-1)]) < 1e-10)
        assert(np.max(np.abs(k2[1:] - 0.75*k2[0:-1] - 0.25*z2[0:-1])) < 1e-10)


def test_GeneralSolution():
    # Create a model object
    ncgm = NeoclassicalGrowth()

    # Create a solution object
    gs = GeneralSolution(ncgm, 2)

    # Test functions that build policy and value functions
    # Note: Not exact because KP/VF computed by a given "rule" and
    #       build_KP/VF come from the regression coefficients fitting
    #       that rule
    assert(np.max(np.abs(gs.KP - gs.build_KP())) < 1e-5)
    assert(np.max(np.abs(gs.VF - gs.build_VF())) < 1e-5)

    # Test function that computes the coeffs given policy and value functions
    k_coeffs, v_coeffs = gs.compute_coefficients(gs.KP, gs.VF)
    assert(np.max(np.abs(k_coeffs - gs.k_coeffs)) < 1e-10)
    assert(np.max(np.abs(v_coeffs - gs.v_coeffs)) < 1e-10)

    # Test that simulate returns same output as jitted function
    shocks = np.random.randn(50)
    ks, zs = gs.simulate(T=50, nburn=0, shocks=shocks)
    kraw, zraw = jit_simulate_ncgm(gs.params, gs.degree, gs.k_coeffs, 50, 0, shocks)
    assert(np.max(np.abs(ks - kraw)) < 1e-10)
    assert(np.max(np.abs(zs - zraw)) < 1e-10)

    # Test that euler errors returns same output as jitted function
    n, w = qe.quad.qnwnorm(10, 0.0, gs.ncgm.sigma**2)
    ee_mean, ee_max = gs.ee_residuals(ksim=ks, zsim=zs)
    eeraw = jit_ee(gs.params, gs.degree, gs.k_coeffs, n, w, ks, zs)
    eeraw_mean, eeraw_max = np.log10(np.mean(eeraw)), np.log10(np.max(eeraw))
    assert(np.max(np.abs(ee_mean - eeraw_mean)) < 1e-10)
    assert(np.max(np.abs(ee_max - eeraw_max)) < 1e-10)


ncgm = NeoclassicalGrowth()
shocks = np.loadtxt("../EE_SHOCKS.csv", delimiter=",")

# Iterate on policy
print("Iterate")
time.sleep(1)
vp = IterateOnPolicy(ncgm, 2)
vp.solve(verbose=True, nskipprint=1)

# VFI
print("VFI")
time.sleep(1)
vp_vfi = VFI(ncgm, 2, vp)
vp_vfi.solve(verbose=True, nskipprint=1, tol=1e-7)
ks, zs = vp_vfi.simulate(10000, 200, shocks=shocks)
mean_ee, max_ee = vp_vfi.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)

# VFI_ECM
print("VFI ECM")
time.sleep(1)
vp_vfiecm = VFI_ECM(ncgm, 2, vp)
vp_vfiecm.solve(verbose=True, nskipprint=1, tol=1e-7)
ks, zs = vp_vfiecm.simulate(10000, 200, shocks=shocks)
mean_ee, max_ee = vp_vfiecm.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)

time.sleep(2)
vp_vfiecm3 = VFI_ECM(ncgm, 3, vp_vfiecm)
vp_vfiecm3.solve(verbose=True, nskipprint=1, tol=1e-7)
ks, zs = vp_vfiecm3.simulate(10000, 200, shocks=shocks)
mean_ee, max_ee = vp_vfiecm3.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)



# VFI_EGM
print("VFI EGM")
time.sleep(1)
vp_vfiegm = VFI_EGM(ncgm, 2, vp)
vp_vfiegm.solve(verbose=True, nskipprint=1)
ks, zs = vp_vfiegm.simulate(10000, 200, 42)
mean_ee, max_ee = vp_vfiegm.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)

# dVFI_ECM
print("dVFI ECM")
time.sleep(1)
vp_dvfiecm = dVFI_ECM(ncgm, 2, vp)
vp_dvfiecm.solve(verbose=True, nskipprint=1)
ks, zs = vp_dvfiecm.simulate(10000, 200, 42)
mean_ee, max_ee = vp_dvfiecm.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)


# PFI
print("PFI")
time.sleep(1)
vp_pfi = PFI(ncgm, 2, vp)
vp_pfi.solve(verbose=True, nskipprint=1)
ks, zs = vp_pfi.simulate(10000, 200, 42)
mean_ee, max_ee = vp_pfi.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)

# PFI ECM
print("PFI ECM")
time.sleep(1)
vp_pfiecm = PFI(ncgm, 2, vp)
vp_pfiecm.solve(verbose=True, nskipprint=1)
ks, zs = vp_pfiecm.simulate(10000, 200, 42)
mean_ee, max_ee = vp_pfiecm.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)

# EE
print("Eul Eq")
time.sleep(1)
vp_ee = EulEq(ncgm, 2, vp)
vp_ee.solve(verbose=True, nskipprint=1)
ks, zs = vp_ee.simulate(10000, 200, 42)
mean_ee, max_ee = vp_ee.ee_residuals(ks, zs, 10)
print(mean_ee, max_ee)

