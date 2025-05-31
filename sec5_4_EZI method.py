# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:40:54 2024

@author: Audren Balon 
"""

import numpy as np

def eigenvalue_zeroing_iteration(A, rank, tol=1e-6, max_iter=1000):
    """
    Eigenvalue Zeroing by Iteration (EZI) algorithm to reduce the rank of matrix A.
    
    Parameters:
    A : np.ndarray
        Input correlation matrix (must be symmetric).
    rank : int
        Desired rank of the output matrix.
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.
    
    Returns:
    B : np.ndarray
        Rank-reduced matrix.
    """   
    # Initialize with input matrix A
    B = np.copy(A)
    prev_B = np.zeros_like(B)
    
    for iteration in range(max_iter):
        # Step 1: Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(B)  # Symmetric matrix eigenvalue decomposition
        
        # Step 2: Zeroing out the smallest eigenvalues to reduce rank
        eigvals_sorted_idx = np.argsort(eigvals)[::-1]  # Sort eigenvalues in descending order
        eigvals[eigvals_sorted_idx[rank:]] = 0  # Set eigenvalues beyond the desired rank to 0
        
        # Reconstruct the matrix with reduced rank
        B = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Step 3: Restore unit diagonal (for correlation matrices)
        np.fill_diagonal(B, 1)
        
        # Check convergence (Frobenius norm)
        diff = np.linalg.norm(B - prev_B, ord='fro')
        if diff < tol:
            print(f'Converged in {iteration + 1} iterations.')
            break
        
        # Update previous matrix
        prev_B = np.copy(B)
    
    return B


# Example usage:
A = np.array([[1.0, 0.8, 0.4],
              [-0.8, 1.0, 0.3],
              [0.4, 0.3, 1.0]])

desired_rank = 2
B_rank_reduced = eigenvalue_zeroing_iteration(A, desired_rank)
print("Rank-Reduced Matrix:")
print(B_rank_reduced)
