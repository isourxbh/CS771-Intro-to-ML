import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.kernel_ridge import KernelRidge


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################
    # --------- Rank-1 factorization: w → u, v ---------
    def _factor_rank1_from_w(w):
        W = np.asarray(w).reshape(33, 33)
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        sigma = S[0]
        u = np.sqrt(sigma) * U[:, 0]
        v = np.sqrt(sigma) * Vh[0, :]
        return np.real_if_close(u), np.real_if_close(v)

    # --------- Convert model vector (33) → delays (32) ---------
    def _model_to_delays(m):
        alpha = np.zeros(32)
        beta  = np.zeros(32)

        # Construct the simplest valid solution
        alpha[0] = m[0]
        alpha[1:] = m[1:32]
        beta[31] = m[32]

        # p-q and r-s differences
        delta = alpha + beta
        gamma = alpha - beta

        # Ensure non-negative delays
        p = np.where(delta >= 0, delta, 0)
        q = np.where(delta >= 0, 0, -delta)
        r = np.where(gamma >= 0, gamma, 0)
        s = np.where(gamma >= 0, 0, -gamma)

        return p, q, r, s

    # ---- Step 1: extract u, v from w ----
    u, v = _factor_rank1_from_w(w)

    # ---- Step 2: convert to delays ----
    a, b, c, d = _model_to_delays(u)
    p, q, r, s = _model_to_delays(v)

    # Use this method to invert a PUF linear model to get back delays
    # w is a single 1089-dim vector (last dimension being the bias term)
    # The output should be eight 32-dimensional vectors

    return a, b, c, d, p, q, r, s
"""
# Load your 10 models
W_all = np.loadtxt("public_mod.txt")   # make sure file is uploaded
print("Loaded models =", W_all.shape)

# Pick one model
w = W_all[0]

# Run decode
a,b,c,d,p,q,r,s = my_decode(w)

# PRINT OUTPUT
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("p =", p)
print("q =", q)
print("r =", r)
print("s =", s) 
"""
