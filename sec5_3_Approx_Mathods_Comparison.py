# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 07:38:40 2025

@author: audre
"""
from matrix_reconstruction import reconstruct_matrix_from_csv
from sec4_1_Semismooth_Newton_with_prec import my_correlationmatrix
import numpy as np
import time 
import matplotlib.pyplot as plt
#Load input matrix of the size of your choice
n = 500
M= reconstruct_matrix_from_csv('unrobustified_sign_correls.csv', n, full=False) 

x_result, y, rank_x, y_tracking, GradNorm, DualObjValue, PrimalObjValue, RelativeDualityGap, exectime, inneriter, trackrank, trackposeig = my_correlationmatrix(M, np.ones((n,1)), tau = 0, tol = 1e-6)
print("Time for Semismooth Method :", exectime['total time'])
print("Rank of the optimum:",rank_x )
def matrix_rank(M, tol=None):
    """
    Calcule le rang numérique d'une matrice M.
    
    Parameters:
    - M : np.ndarray, matrice d'entrée
    - tol : float, tolérance pour considérer une valeur singulière comme non nulle.
            Si None, tol est calculée par défaut comme max(M.shape) * eps * max(singular_values)
    
    Returns:
    - rank : int, rang numérique de la matrice M
    """
    s = np.linalg.svd(M, compute_uv=False)
    if tol is None:
        tol = max(M.shape) * np.finfo(float).eps * np.max(s)
    rank = np.sum(s > tol)
    return rank
###############################################################################
### Principal Components Analysis
###############################################################################
tic = time.time()
# Step 1: Eigenvalue decomposition
eigvals, eigvecs = np.linalg.eigh(M)

# Step 2: Set negative eigenvalues to zero
eigvals = np.maximum(eigvals, 0)  # Replace negative eigenvalues with zero

# Compute Z = P * Lambda^{1/2}
Lambda_sqrt = np.sqrt(np.diag(eigvals))  # Square root of non-negative eigenvalues
J = eigvecs @ Lambda_sqrt  # Z = P * Lambda^{1/2}

# Step 3: Normalize the rows of Z to obtain X
X = np.array([z_row / np.linalg.norm(z_row) for z_row in J])

# Step 4: Compute M = X * X^T
M_PCA = X @ X.transpose()
tac = time.time() - tic
print("Time for PCA processing:", tac )
# Display the matrix M (optional)
print("Optimum Approximation by PCA:\n")
print("Distance to the optimum for PCA :", np.linalg.norm(M_PCA-x_result, 'fro'))
print("Rank of the approximation by PCA :", matrix_rank(M_PCA))
k = 15
flag = 0
###############################################################################
### Subspace Restriction Method
### Primal Method 
###############################################################################
import cvxpy as cp
import numpy as np

Sigma_plus = np.diag(eigvals[:k])
e = np.ones(n)  # Vector of diagonal constraints, here equal to 1 for each element
U = eigvecs[:, :k]

# Define the optimization variable Z (symmetric matrix)
Z = cp.Variable((k, k), symmetric=True)

# Define the objective: minimize the Frobenius norm
objective = cp.Minimize(0.5 * cp.norm(Z - Sigma_plus, "fro")**2)

# Define the constraints
constraints = [
    Z >> 0,  # Z must be positive semidefinite
    cp.diag(U @ Z @ U.T) == e  # The diagonal elements of U Z U^T must equal e
]

# Define the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the problem using the SCS solver
tip = time.time()
problem.solve(solver=cp.SCS)
top = time.time() - tip

# Display the solution
if problem.status == cp.OPTIMAL:
    print("Optimal solution found:")
    print("Optimized matrix Z:")
    print(Z.value)
else:
    flag = 1
    print("The problem does not have an optimal solution.")

if flag == 0:
    X_SRM = U @ Z.value @ U.T

    frobenius_approx1 = np.linalg.norm(X_SRM - x_result, 'fro')
    print("Time for SRM processing:", top)
    print("Distance to the optimum for SRM:", frobenius_approx1)
    print("Rank of the SRM approximation:", matrix_rank(X_SRM))
###############################################################################
### Scaled-Subspace Restriction Method
###############################################################################
def plot_spg_history(f_hist, step_norms, alphas):
    iterations = range(len(f_hist))

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(iterations, f_hist, marker='o')
    plt.title("Objective function value")
    plt.xlabel("Iteration")
    plt.ylabel("f(Z)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(range(len(step_norms)), step_norms, marker='o', color='orange')
    plt.title("Step norm")
    plt.xlabel("Iteration")
    plt.ylabel("||d||_F")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(len(alphas)), alphas, marker='o', color='green')
    plt.title("Step size alpha")
    plt.xlabel("Iteration")
    plt.ylabel("alpha")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def project_to_SPSD(Z):
    """Projection on symmetric positive semidefinite cone."""
    Z_sym = (Z + Z.T) / 2
    eigvals, eigvecs = np.linalg.eigh(Z_sym)
    eigvals_proj = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(eigvals_proj) @ eigvecs.T

def f(Z, M, U):
    UZUt = U @ Z @ U.T
    d = 1.0 / np.sqrt(np.diag(UZUt))
    D = np.diag(d)
    DUZUtD = D @ UZUt @ D
    return np.linalg.norm(M - DUZUtD, 'fro')**2

def grad_f(Z, M, U):
    UZUt = U @ Z @ U.T
    d = 1.0 / np.sqrt(np.diag(UZUt))
    D = np.diag(d)
    DUZUtD = D @ UZUt @ D
    M_minus = M - DUZUtD
    D3 = D @ D @ D
    # Computing terms
    term1 = U.T @ np.diag(np.diag(UZUt @ D @ M_minus @ D3)) @ U
    term2 = 2 * (U.T @ D @ M_minus @ D @ U)
    term3 = U.T @ np.diag(np.diag(M_minus @ D @ UZUt @ D3)) @ U
    return term1 - term2 + term3

def spg_method(M, U, Z0, max_iter=150, tol=1e-6, M_gll=10, gamma=1e-4, sigma1=0.1, sigma2=0.9):
    Z = Z0.copy()
    f_hist = [f(Z, M, U)]
    grad = grad_f(Z, M, U)
    
    lmbda = 1.0

    step_norms = []
    alphas = []

    for k in range(max_iter):
        Z_temp = project_to_SPSD(Z - lmbda * grad)
        d = Z_temp - Z
        
        norm_d = np.linalg.norm(d, 'fro')
        if norm_d < tol:
            print(f'Converged at iteration {k}')
            break
        
        fmax = max(f_hist[-M_gll:]) if len(f_hist) >= M_gll else max(f_hist)
        alpha = 1.0
        
        while True:
            Z_new = project_to_SPSD(Z + alpha * d)
            f_new = f(Z_new, M, U)
            delta = np.sum(grad * d)
            
            if f_new <= fmax + gamma * alpha * delta:
                break
            
            numerator = -0.5 * (alpha**2) * delta
            denominator = f_new - f(Z, M, U) - alpha * delta
            if denominator == 0:
                alpha = alpha / 2
            else:
                alpha_tmp = numerator / denominator
                if sigma1 <= alpha_tmp <= sigma2 * alpha:
                    alpha = alpha_tmp
                else:
                    alpha = alpha / 2
            
            if alpha < 1e-10:
                break
        
        Z = Z_new
        grad = grad_f(Z, M, U)
        f_val = f(Z, M, U)
        f_hist.append(f_val)

        s = alpha * d
        y = grad - grad_f(Z - s, M, U)
        s_flat = s.flatten()
        y_flat = y.flatten()
        denom = np.dot(s_flat, y_flat)
        if denom > 1e-10:
            lmbda = np.dot(s_flat, s_flat) / denom
        else:
            lmbda = 1.0
        
        print(f"Iter {k}: objective = {f_val:.6e}, step norm = {norm_d:.2e}, alpha = {alpha:.2e}")

        step_norms.append(norm_d)
        alphas.append(alpha)

    return Z, f_hist, step_norms, alphas


def low_rank_psd_projection(M, k, eigvecs, eigvals):
    """
    Keeps the top k strictly positive eigenvalues.
    If there are fewer than k, raises an error.
    """
    # Sort eigenvalues and eigenvectors in decreasing order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Filter strictly positive eigenvalues
    positive_idx = eigvals > 0
    eigvals_pos = eigvals[positive_idx]
    eigvecs_pos = eigvecs[:, positive_idx]

    if len(eigvals_pos) < k:
        raise ValueError(f"Number of strictly positive eigenvalues ({len(eigvals_pos)}) is less than k={k}")

    eigvals_k = eigvals_pos[:k]
    eigvecs_k = eigvecs_pos[:, :k]

    return eigvals_k, eigvecs_k



eigvals_k, eigvecs_k = low_rank_psd_projection(M, k, eigvecs, eigvals)

M_k = eigvecs_k @ np.diag(eigvals_k) @ eigvecs_k.T

# Initial guess for Z
Z0 = np.diag(eigvals_k)

# Run SPG
t0=time.time()
Z_sol, f_hist, step_norms, alphas = spg_method(M_k, eigvecs_k, Z0)
plot_spg_history(f_hist, step_norms, alphas)
tend = time.time()

def compute_D(U, Z):
    UZUt = U @ Z @ U.T
    diag_vals = np.diag(UZUt)
    if np.any(diag_vals <= 0):
        raise ValueError("Non-positive values on the diagonal encountered in compute_D")
    D = np.diag(1.0 / np.sqrt(diag_vals))
    return D

# Evaluate final approximation
D_final = compute_D(eigvecs_k, Z_sol)
M_approx = D_final @ (eigvecs_k @ Z_sol @ eigvecs_k.T) @ D_final
objective = np.linalg.norm(M_k - M_approx, 'fro') 

print(f"\nFinal Objective Value: {objective:.6e}")
print("Time for approx", tend-t0)
error2 = np.linalg.norm(x_result - M_approx, 'fro')
print(f"\nDistance to the optimum for Scaled-SRM: {error2:.6e}")

print(matrix_rank(M_approx))



print("\n=== Target Rank ===")
print(f"Target Rank (k): {k}")

print("\n=== Semismooth Newton Method (Exact) ===")
print(f"Execution Time: {exectime['total time']:.2f} seconds")
print(f"Rank of the Optimum Solution: {rank_x}")

print("\n=== PCA Approximation ===")
print(f"PCA Processing Time: {tac:.2f} seconds")
print(f"Distance to Optimum: {np.linalg.norm(M_PCA - x_result, 'fro'):.6e}")
print(f"Rank of PCA Approximation: {matrix_rank(M_PCA)}")

print("\n=== Scaled-SRM Approximation ===")
print(f"Execution Time: {tend - t0:.2f} seconds")
print(f"Distance to Optimum: {error2:.6e}")
print(f"Rank of Scaled-SRM Approximation: {matrix_rank(M_approx)}")
if flag == 0 : 
    print("\n=== SRM Approximation ===")
    print(f"Time for SRM processing:{top:.2f}" )
    print(f"Distance to Optimum: {frobenius_approx1:.6e}")
    print(f"Rank of SRM Approximation: {matrix_rank(X_SRM)}")


ks = np.arange(10, 130, 10)  # Choose the values of k to test

errors = []
times = []
ranks = []

for k in ks:
    # Project M onto rank k
    eigvals_k, eigvecs_k = low_rank_psd_projection(M, k, eigvecs, eigvals)
    M_k = eigvecs_k @ np.diag(eigvals_k) @ eigvecs_k.T
    Z0 = np.diag(eigvals_k)

    # Run the SPG method
    t0 = time.time()
    Z_sol, f_hist, step_norms, alphas = spg_method(M_k, eigvecs_k, Z0)
    t1 = time.time()

    # Final approximation
    D_final = compute_D(eigvecs_k, Z_sol)
    M_approx = D_final @ (eigvecs_k @ Z_sol @ eigvecs_k.T) @ D_final

    # Store the measurements
    exec_time = t1 - t0
    error = np.linalg.norm(x_result - M_approx, 'fro')
    rank_approx = np.linalg.matrix_rank(M_approx)

    times.append(exec_time)
    errors.append(error)
    ranks.append(rank_approx)

plt.figure(figsize=(16, 5))

# Frobenius Error
plt.subplot(1, 3, 1)
plt.plot(ks, errors, marker='o', color='blue', label='Scaled-SRM Error')
plt.title('Frobenius Error vs Target Rank k')
plt.xlabel('Target Rank k')
plt.ylabel(r'$||x_{\mathrm{result}} - M_{\mathrm{approx}}||_F$')
plt.grid(True)

# Execution time with Semismooth reference line
plt.subplot(1, 3, 2)
plt.plot(ks, times, marker='o', color='green', label='Scaled-SRM Time')
plt.axhline(y=exectime['total time'], color='black', linestyle='--', label='Semismooth Time')
plt.axhline(y=tac, color='red', linestyle='--', label='PCA Time')
plt.title('Execution Time vs Target Rank k')
plt.xlabel('Target Rank k')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)

# Effective rank with reference lines for ground-truth and PCA
plt.subplot(1, 3, 3)
plt.plot(ks, ranks, marker='o', color='red', label='Rank of $M_{approx}$')
plt.axhline(y=rank_x, color='black', linestyle='--', label='Rank of $x_{result}$')
plt.axhline(y=matrix_rank(M_PCA), color='red', linestyle='--', label='Rank of $M_{PCA}$')
plt.title('Effective Rank vs Target Rank k')
plt.xlabel('Target Rank k')
plt.ylabel('Effective Rank')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

