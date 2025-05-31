# -*- coding: utf-8 -*-
"""
Created on Sat May 24 21:26:11 2025

@author: audre
"""

import numpy as np
from scipy.optimize import brentq
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed  # For parallelization

def secular_function(x, d, v, rho):
    # Simplified example, replace with your actual function
    # Watch out for division by zero when x == d[i]
    return 1 + rho * np.sum(v**2 / (d - x))

def solve_secular_equation_interval(a, b, d, v, rho, tol=1e-12):
    fa = secular_function(a, d, v, rho)
    fb = secular_function(b, d, v, rho)
    if np.sign(fa) == np.sign(fb):
        raise ValueError(f"Function has same signs at interval boundaries: f({a})={fa}, f({b})={fb}")
    root = brentq(secular_function, a, b, args=(d, v, rho), xtol=tol)
    return root

def find_root_with_expansion(a, b, d, v, rho, tol=1e-12):
    """First try the interval [a,b], otherwise expand until finding a sign change"""
    try:
        return solve_secular_equation_interval(a, b, d, v, rho, tol)
    except ValueError:
        # Try expanding the interval gradually
        for shift in np.linspace(0.1, 10, 100):
            try:
                root = solve_secular_equation_interval(a - shift, b + shift, d, v, rho, tol)
                return root
            except ValueError:
                continue
        raise RuntimeError(f"Cannot find root in interval [{a},{b}] even after expansion")

def solve_secular_equation(d, v, rho, tol=1e-12, n_jobs=-1):
    n = len(d)
    intervals = []
    for i in range(n - 1):
        a, b = d[i], d[i+1]
        if rho > 0:
            a += tol
            b -= tol
        else:
            a -= tol
            b += tol
        intervals.append((a, b))

    # Find roots in the intervals in parallel
    roots = Parallel(n_jobs=n_jobs)(
        delayed(find_root_with_expansion)(a, b, d, v, rho, tol) for (a, b) in intervals
    )

    # Find the last root outside the bounds of d (sequential)
    if rho > 0:
        a = d[-1] + tol
        b = d[-1] + 100
    else:
        a = d[0] - 100
        b = d[0] - tol

    fa = secular_function(a, d, v, rho)
    fb = secular_function(b, d, v, rho)
    if np.sign(fa) == np.sign(fb):
        found = False
        for shift in np.linspace(0.1, 1000, 1000):
            a_new = a - shift if rho > 0 else a + shift
            b_new = b + shift if rho > 0 else b - shift
            try:
                fa = secular_function(a_new, d, v, rho)
                fb = secular_function(b_new, d, v, rho)
                if np.sign(fa) != np.sign(fb):
                    a, b = a_new, b_new
                    found = True
                    break
            except:
                continue
        if not found:
            raise RuntimeError("Cannot find suitable interval for last root")

    last_root = brentq(secular_function, a, b, args=(d, v, rho), xtol=tol)
    roots.append(last_root)

    eigenvalues = np.array(roots)
    eigenvalues.sort()
    return eigenvalues

def compute_eigenvectors(d, v, eigenvalues, rho):
    n = len(d)
    Q = np.zeros((n, n))
    for j, lam in enumerate(eigenvalues):
        q_j = v / (d - lam)
        q_j /= np.linalg.norm(q_j)
        Q[:, j] = q_j
    return Q

def rank_one_update_eigendecomposition(d, v, rho, n_jobs=1):
    d = np.asarray(d)
    v = np.asarray(v)
    assert d.ndim == 1 and v.ndim == 1 and d.shape == v.shape
    
    eigenvalues = solve_secular_equation(d, v, rho, n_jobs=n_jobs)
    Q = compute_eigenvectors(d, v, eigenvalues, rho)
    return eigenvalues, Q

# --- Function to apply k successive rank-1 updates ---
def apply_k_rank_one_updates(d, Vs, rhos, n_jobs=1):
    """
    d: initial diagonal vector (shape n)
    Vs: list or array (k x n) of vectors v_i
    rhos: vector (k) of scalars rho_i
    Returns the eigenvalues and eigenvectors after k updates
    """
    current_d = d.copy()
    current_Q = np.eye(len(d))  # Initially the identity matrix
    
    for i in range(len(rhos)):
        # Project v_i into the current basis
        v_i = Vs[i]
        rho_i = rhos[i]
        
        # Transform v_i in the current Q basis
        v_proj = current_Q.T @ v_i
        
        # Perform the rank-1 update decomposition
        lambdas, Q_update = rank_one_update_eigendecomposition(current_d, v_proj, rho_i, n_jobs=n_jobs)
        
        # Update current_d and current_Q
        current_d = lambdas
        current_Q = current_Q @ Q_update
        
    return current_d, current_Q


# --- Test 1 ---
if __name__ == "__main__":
    n = 200
    k = 100

    np.random.seed(0)
    d = np.linspace(1, n, n)
    Vs = np.random.randn(k, n)
    rhos = np.random.uniform(0.1, 1.0, size=k)

    # Method 1 time: k successive rank-1 updates
    start = time.time()
    eigenvals_1, eigenvecs_1 = apply_k_rank_one_updates(d, Vs, rhos, n_jobs=-1)
    time_method_1 = time.time() - start
    print(f"Method 1 (successive rank-1 updates): {time_method_1:.4f}s")

    # Method 2 time: direct decomposition of D + sum rho_i v_i v_i^T
    M = np.diag(d)
    for i in range(k):
        M += rhos[i] * np.outer(Vs[i], Vs[i])

    start = time.time()
    eigenvals_2, eigenvecs_2 = np.linalg.eigh(M)
    time_method_2 = time.time() - start
    print(f"Method 2 (direct decomposition): {time_method_2:.4f}s")

    # Check error between eigenvalues
    err_vals = np.linalg.norm(np.sort(eigenvals_1) - np.sort(eigenvals_2))
    print(f"Error on eigenvalues norm: {err_vals:.2e}")

    # Check reconstruction
    M_recon_1 = eigenvecs_1 @ np.diag(eigenvals_1) @ eigenvecs_1.T
    err_recon_1 = np.linalg.norm(M - M_recon_1)
    print(f"Reconstruction error method 1: {err_recon_1:.2e}")

    M_recon_2 = eigenvecs_2 @ np.diag(eigenvals_2) @ eigenvecs_2.T
    err_recon_2 = np.linalg.norm(M - M_recon_2)
    print(f"Reconstruction error method 2: {err_recon_2:.2e}")

    # Display time comparison
    plt.bar(["Rank-1 updates (k={})".format(k), "Direct decomposition"], [time_method_1, time_method_2])
    plt.ylabel("CPU time (seconds)")
    plt.title("CPU time comparison")
    plt.show()


def apply_k_rank_one_updates_test(d, Vs, rhos, n_jobs=-1):
    """
    Simulates Method 1: successive rank-1 updates of the spectral decomposition.
    This is a dummy version to make the script run;
    replace it with your actual implementation.
    """
    k = Vs.shape[0]
    # For testing purposes, just return the direct decomposition (slow!)
    M = np.diag(d)
    for i in range(k):
        M += rhos[i] * np.outer(Vs[i], Vs[i])
    eigenvals, eigenvecs = np.linalg.eigh(M)
    return eigenvals, eigenvecs


# --- Test 2 ---
if __name__ == "__main__":
    np.random.seed(0)
    k = 50  # Fixed rank

    ns = [200, 400, 800, 1200, 1600]  # Values of n to test
    times_method_1 = []
    times_method_2 = []

    for n in ns:
        d = np.linspace(1, n, n)
        Vs = np.random.randn(k, n)
        rhos = np.random.uniform(0.1, 1.0, size=k)

        # Method 1
        start = time.time()
        eigenvals_1, eigenvecs_1 = apply_k_rank_one_updates_test(d, Vs, rhos, n_jobs=-1)
        times_method_1.append(time.time() - start)

        # Method 2
        M = np.diag(d)
        for i in range(k):
            M += rhos[i] * np.outer(Vs[i], Vs[i])

        start = time.time()
        eigenvals_2, eigenvecs_2 = np.linalg.eigh(M)
        times_method_2.append(time.time() - start)

        print(f"n={n} | Method 1: {times_method_1[-1]:.3f}s | Method 2: {times_method_2[-1]:.3f}s")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(ns, times_method_1, 'o-', label=f"Rank-1 updates (k={k})")
    plt.plot(ns, times_method_2, 's-', label="Direct decomposition")
    plt.xlabel("Dimension n")
    plt.ylabel("CPU time (seconds)")
    plt.title("CPU Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    
# --- Test 3 ---
if __name__ == "__main__":
    np.random.seed(0)
    n = 1000  # Fixed matrix size
    ks = [10, 20, 50, 100, 200, 300, 400, 500]  # Values of k to test

    times_method_1 = []
    times_method_2 = []

    for k in ks:
        print(f"Testing for k = {k}")
        d = np.linspace(1, n, n)
        Vs = np.random.randn(k, n)
        rhos = np.random.uniform(0.1, 1.0, size=k)

        # Method 1
        start = time.time()
        eigenvals_1, eigenvecs_1 = apply_k_rank_one_updates(d, Vs, rhos, n_jobs=-1)
        times_method_1.append(time.time() - start)

        # Method 2
        M = np.diag(d)
        for i in range(k):
            M += rhos[i] * np.outer(Vs[i], Vs[i])
        start = time.time()
        eigenvals_2, eigenvecs_2 = np.linalg.eigh(M)
        times_method_2.append(time.time() - start)

        print(f"    Method 1: {times_method_1[-1]:.3f}s | Method 2: {times_method_2[-1]:.3f}s")

    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(ks, times_method_1, 'o-', label="Rank-1 updates")
    plt.plot(ks, times_method_2, 's-', label="Direct decomposition")
    plt.xlabel("Rank k (Number of rank-one updates)")
    plt.ylabel("CPU time (seconds)")
    plt.title(f"CPU times (n = {n})")
    plt.legend()
    plt.grid(True)
    plt.show()
