# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:35:39 2025

@author: Audren
"""

import numpy as np

# Step 1: Initial symmetric matrix
D = np.array([[2, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=float)

# Diagonalization of D
lam, U = np.linalg.eigh(D)

# Step 2: Diagonal perturbation
delta_y = np.array([0.4, 0.0, -0.2])
D_perturbed = D + np.diag(delta_y)

# Step 3: Exact diagonalization of D + diag(delta_y)
lam_true, U_true = np.linalg.eigh(D_perturbed)

# Step 4: First-order correction
def first_order_correction(lam, U, delta_y):
    # Clean outliers
    delta_y_cleaned = np.copy(delta_y)
    print(delta_y_cleaned[:, None])
    print(delta_y_cleaned[:, None].shape)
    print(U.shape)
    # Compute A matrix: A_ij = u_i^T ΔM u_j
    A = delta_y_cleaned[:, None] * U
    print(U)
    print(A)
    A = np.dot(U.T, A)
    
    print("Matrix of pertubations:\n", A)
    print("Check if the optimized computation above is identical to the matrix product one :\n")
    print(U.T @ np.diag(delta_y_cleaned) @ U)

    lam = lam.flatten()
    diffs = lam[:, None] - lam[None, :]
    np.fill_diagonal(diffs, np.inf)

    print("Weight matrix for pertubation of eigenvectors :\n" ,diffs)

    # First-order correction terms
    correction = A.T / diffs
    print("Full correction matrix for the eigenvectors: \n")
    print(correction)

    # Apply correction to eigenvectors
    U_corr = U + np.dot(U, correction)
    print("Vectorized computation of approximated eigenvector 1 : \n", U_corr[:, 0])
    # Verification of corrected eigenvector ũ_0
    print("Computation of approximated eigenvector 1: \n", U[:, 0] + correction[1, 0] * U[:, 1] + correction[2, 0] * U[:, 2])

    # Apply correction to eigenvalues
    lam_corr = lam + np.diag(A)

    return lam_corr, U_corr

# Run the correction
lam_corr, U_corr = first_order_correction(lam, U, delta_y)

# Display results
print("Original eigenvalues:", lam.round(4))
print("Approximated eigenvalues:", lam_corr.round(4))

print("\nApproximated eigenvectors:")
print(U_corr.round(4))
print("\nReal eigenvectors:")
print(U_true.round(4))

# Compare approximated and true eigenvectors (insensitive to sign)
similarity_matrix = np.abs(U_corr.T @ U_true)
angles_rad = np.arccos(np.clip(np.abs(np.sum(U_true * U_corr, axis=0)), 0, 1))
angles_deg = np.degrees(angles_rad)
print("Angles between real and approximated eigenvectors:\n", angles_deg)
print("Are eigenvectors orthonormal? U^T U should be close to identity:")
print(np.round(similarity_matrix, 4))

