import numpy as np

np.random.seed(61089)
shocks = np.random.randn(10200)

np.savetxt("EE_SHOCKS.csv", shocks, delimiter=",")

