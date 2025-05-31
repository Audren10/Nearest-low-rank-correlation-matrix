# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 09:03:36 2025

@author: audre
"""

import numpy as np
import time

def naive_qr(A, max_iter=50000, tol=1e-2, Q_init=None):
    """ Naive implementation of the QR algorithm to compute eigenvalues and eigenvectors.
    
    Parameters:
    A : Input matrix (square, symmetric for PSD projection)
    max_iter : Maximum number of iterations
    tol : Convergence tolerance (maximum difference between iterations)
    Q_init : Optional, initial Q for warm start

    Returns:
    eigenvalues : Approximated eigenvalues
    Q : Matrix whose columns are the eigenvectors
    max_attained : Boolean indicating if the maximum number of iterations was reached
    """
    n = A.shape[0]
    A_k = A.copy()
    Q = np.eye(n) if Q_init is None else Q_init
    max_attained = False

    for i in range(max_iter):
        Q_k, R_k = np.linalg.qr(A_k)  # QR decomposition
        A_k = R_k @ Q_k  
        Q = Q @ Q_k  

        # Stopping criterion: check if A_k is diagonal
        off_diag_norm = np.linalg.norm(A_k - np.diag(np.diag(A_k))) / np.linalg.norm(A_k)
        if i == max_iter - 1:
            max_attained = True
        if off_diag_norm < tol:
            break
    
    eigenvalues = np.diag(A_k)
    return eigenvalues, Q, max_attained


def naive_qr_shift(A, max_iter=10000, tol=1e-1, Q_init=None):
    """ QR algorithm with shift to compute eigenvalues and eigenvectors.
    
    Parameters:
    A : Input matrix (square, symmetric for PSD projection)
    max_iter : Maximum number of iterations
    tol : Convergence tolerance (maximum difference between iterations)
    Q_init : Optional, initial Q for warm start

    Returns:
    eigenvalues : Approximated eigenvalues
    Q : Matrix whose columns are the eigenvectors
    max_attained : Boolean indicating if the maximum number of iterations was reached
    """
    n = A.shape[0]
    A_k = A.copy()
    Q = np.eye(n) if Q_init is None else Q_init
    max_attained = False

    for i in range(max_iter):
        # Choose a shift (often the last diagonal entry)
        mu = A_k[-1, -1]

        # Subtract the shift
        A_shifted = A_k - mu * np.eye(n)

        # QR decomposition
        Q_k, R_k = np.linalg.qr(A_shifted)

        # Reconstruct matrix with shift
        A_k = R_k @ Q_k + mu * np.eye(n)
        
        # Update Q
        Q = Q @ Q_k  

        # Stopping criterion
        off_diag_norm = np.linalg.norm(A_k - np.diag(np.diag(A_k))) / np.linalg.norm(A_k)
        if i == max_iter - 1:
            max_attained = True
        if off_diag_norm < tol:
            break
    
    eigenvalues = np.diag(A_k)
    return eigenvalues, Q, max_attained


def householder(A):
    """ Applies Householder transformation to reduce a symmetric matrix to tridiagonal form. """
    n = A.shape[0]
    Q = np.eye(n)

    for k in range(n - 2):
        # Create vector v for Householder transformation
        x = A[k+1:n, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) if x[0] >= 0 else -np.linalg.norm(x)
        v = x - e1
        v = v / np.linalg.norm(v)

        # Apply Householder transformation
        A[k+1:n, k:n] -= 2 * np.outer(v, np.dot(v.T, A[k+1:n, k:n]))
        A[:, k+1:n] -= 2 * np.outer(A[:, k+1:n].dot(v), v)
        Q[:, k+1:n] -= 2 * np.outer(Q[:, k+1:n].dot(v), v)

    return A, Q


def naive_qr_householder(A, max_iter=50000, tol=1e-2, Q_init=None):
    """ QR algorithm with Householder reduction to compute eigenvalues and eigenvectors. 
    
    Parameters:
    A : Input matrix (square, symmetric)
    max_iter : Maximum number of iterations
    tol : Convergence tolerance
    Q_init : Optional initial orthogonal matrix

    Returns:
    eigenvalues : Approximated eigenvalues
    Q : Matrix of eigenvectors
    max_attained : Boolean indicating if iteration limit was reached
    """
    n = A.shape[0]
    A_k = A.copy()
    Q = np.eye(n) if Q_init is None else Q_init
    max_attained = False

    # Apply Householder to make A tridiagonal
    A_k, Q_householder = householder(A_k)
    Q = Q @ Q_householder  # Update Q

    for i in range(max_iter):
        # QR decomposition
        Q_k, R_k = np.linalg.qr(A_k)
        
        # Update A_k
        A_k = R_k @ Q_k

        # Update Q
        Q = Q @ Q_k

        # Stopping criterion
        off_diag_norm = np.linalg.norm(A_k - np.diag(np.diag(A_k))) / np.linalg.norm(A_k)
        if i == max_iter - 1:
            max_attained = True
        if off_diag_norm < tol:
            break

    eigenvalues = np.diag(A_k)
    return eigenvalues, Q, max_attained


# Quick test
if __name__ == "__main__":
    n = 50  # Matrix size
    A = np.random.randn(n, n)
    A = (A + A.T) / 2  # Make the matrix symmetric

    tic = time.time()
    eigenvalues, eigenvectors, max_attained = naive_qr(A)
    print("Time for naive QR:", time.time() - tic)
    print("Max iterations reached:", max_attained)

    toc = time.time()
    eigenvalues, eigenvectors, max_attained = naive_qr_householder(A)
    print("Time for QR with Householder:", time.time() - toc)
    print("Max iterations reached:", max_attained)

    tac = time.time()
    eigenvalues, eigenvectors, max_attained = naive_qr_shift(A)
    print("Time for QR with shift:", time.time() - tac)
    print("Max iterations reached:", max_attained)

    eig, _ = np.linalg.eigh(A)
    print("First 10 computed eigenvalues:", sorted(eigenvalues)[:10])
    print("First 10 true eigenvalues:", sorted(eig)[:10])
