import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.kernel_ridge import KernelRidge

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

# ===============================
# DIRECT LOAD (FILES IN /content)
# ===============================
#X_trn = np.loadtxt("public_x_trn.txt")
#X_tst = np.loadtxt("public_x_tst.txt")

#Y_trn = np.loadtxt("public_y_trn.txt")
#Y_tst = np.loadtxt("public_y_tst.txt")

#Z_trn = np.loadtxt("public_Z_trn.txt")
#Z_tst = np.loadtxt("public_Z_tst.txt")


#print("Loaded all datasets successfully!")

#print("Shapes:")
"""
print("X_trn =", X_trn.shape)
print("X_tst =", X_tst.shape)
print("Y_trn =", Y_trn.shape)
print("Y_tst =", Y_tst.shape)
print("Z_trn =", Z_trn.shape)
print("Z_tst =", Z_tst.shape)

print("\nSample rows:")
print("X_trn:", X_trn[:5])
print("Z_trn:", Z_trn[:5])
print("Y_trn:", Y_trn[:5])

# reshape X into (n, 1)
X_trn = X_trn.reshape(-1, 1)
X_tst = X_tst.reshape(-1, 1)

# combine [x , z1 , z2]
XZ_trn = np.hstack([X_trn, Z_trn])
XZ_tst = np.hstack([X_tst, Z_tst])

print("XZ_trn shape:", XZ_trn.shape)
print("XZ_tst shape:", XZ_tst.shape)
"""

################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################
    # Kernel over z
    Kz = polynomial_kernel(Z1, Z2, degree=3, coef0=1)

    # Outer product x1 * x2
    X_outer = X1 @ X2.T

    # Final Gram matrix
    G = X_outer * Kz + 1   # +1 for the bias

    # Use this method to compute Gram matrices for your proposed kernel
    # Your kernel matrix will be used to train a kernel ridge regressor

    return G
"""
K_train = my_kernel(X_trn, Z_trn, X_trn, Z_trn)

model = KernelRidge(kernel="precomputed")
model.fit(K_train, Y_trn)

K_test = my_kernel(X_tst, Z_tst, X_trn, Z_trn)

print("Test R2 =", model.score(K_test, Y_tst))
"""


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