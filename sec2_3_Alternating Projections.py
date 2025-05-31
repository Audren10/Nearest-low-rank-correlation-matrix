# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:41:45 2024

@author: Audren Balon
"""
# inspired from the matlab code of Higham available in his github 
# https://github.com/higham/anderson-accel-ncm

###############################################################################
############Alternating methods, projections, and accelerations ###############
###############################################################################
import scipy as sp
import numpy as np 
from numpy.linalg import solve
from scipy.linalg import sqrtm
import time
## Alternating Projection by Higham (2002)


def alternating_proj(M, W, itmax = 1000, eps = 1e-6, method = 'positive only'): 
    """
    Compute the nearest correlation matrix 
    Parameters 
    -----------
    M: the input matrix 
    W : the weigting matrix use din the norm of the nearest correlation matrix problem 
    itmax : max number of iterate allowed
    eps : tolerance used for the stopping criteria : ensures that the solution lies in the intersection of both sets
    
    Returns 
    --------
    The nearest correlation matrix using the alternating projection method
    """
    #Initialization 
    n = len(M[0:])
    deltaS = np.zeros(n)
    Y = M
    X = np.zeros(n)
    inverse_W = np.linalg.inv(W)
    for i in range(itmax): 
        R = Y - deltaS  #Dijkstra correction for projection onto the cone S_+^n 
        X_old = X
        X = proj_S(R, W, inverse_W, method )
        deltaS = X - R 
        Y_old = Y 
        Y = proj_K(X, inverse_W)   #No correction needed since K is an affine subspace
        dist = max(np.linalg.norm(X-X_old, ord = np.inf)/ np.linalg.norm(X, ord = np.inf), np.linalg.norm(Y-Y_old, ord = np.inf)/ np.linalg.norm(Y, ord = np.inf), np.linalg.norm(X-Y, ord = np.inf)/ np.linalg.norm(Y, ord = np.inf))
        #dist =np.linalg.norm(X-Y, 'fro')
        if dist < eps  : 
            return Y, i+1, dist  
    raise Exception("No convergence in the specified number of iterations")


def proj_S(A, W, invW, method): 
    """
    Parameters
    ----------
    matrix A 

    Returns
    -------
    The projection of A onto the cone of symmetric positive semidefinite matrices 

    """
    B = sqrtm(W)@A@sqrtm(W)
    eigvals, eigvecs = decomposition_eig(B, method)
    eigvals_proj = np.maximum(eigvals, 0)
    C = eigvecs @ np.diag(eigvals_proj) @ eigvecs.T 
    D = sqrtm(invW) @ C @ sqrtm(invW)
    return (D + D.T)
    
def proj_K(A, invW): 
    """
    Parameters
    ----------
    matrix A 

    Returns
    -------
    The projection of A onto K, the affine subspace of symmetric matrices with unit diagonal 

    """
    n = len(A[0:])
    if np.allclose(invW, np.eye(n)): 
        return A - np.diag(np.diag(A)) + np.eye(n)
    else: 
        #Compute Theta
        inv2W = invW * invW
        theta = solve(inv2W, np.diag(A-np.eye(n)))
        
        return A - invW @ np.diag(theta) @ invW 
    
### The biggest challenge for large scale matrices is computing the eigenvalues and eigenvectors
### in a reasonable CPU time (reasonable is defined according to matrices of size 18895 and more)
def decomposition_eig(A,method): 
    if method == 'classic': 
        return np.linalg.eigh(A)
    if method == 'positive only': 
        return sp.linalg.eigh(A, subset_by_value=(1e-6, np.inf))
    else : 
        raise ValueError("This method is not covered.")

def reconstruct_matrix_from_csv(file_path, matrix_size, full= False):
    """
    Reconstructs a full symmetric correlation matrix from a CSV file containing the lower-left part.

    Parameters:
        file_path (str): Path to the CSV file containing the lower-left part of the matrix.
        matrix_size (int): The size of the full square matrix (number of rows/columns).

    Returns:
        np.ndarray: The reconstructed symmetric correlation matrix.
    """
    if full == True : 
        with open(file_path, 'r') as f:
            data = f.read().strip()
        values = np.array(list(map(float, data.split(',')))) 
    else: 
        # Calculate the number of values to read
        num_values = matrix_size * (matrix_size - 1) // 2
    
        # Load the required number of values from the CSV file into a 1D NumPy array
        values = []
        with open(file_path, 'r') as f:
            for line in f:
                for value in line.strip().split(','):
                    values.append(float(value))
                    if len(values) == num_values:
                        break
                if len(values) == num_values:
                    break
        values = np.array(values)
    
    # Initialize an empty matrix of the given size
    matrix = np.zeros((matrix_size, matrix_size))

    # Fill in the lower triangular part of the matrix from the file values
    index = 0
    for j in range(0, matrix_size-1):
        for i in range(j+1, matrix_size):
            matrix[i, j] = values[index]
            index += 1
    # Copy the lower triangular part to the upper triangular part to make it symmetric
    matrix += matrix.T

    # Fill the diagonal with ones
    np.fill_diagonal(matrix, 1.0)

    return matrix
n = 3000
tic = time.time()
mat_input = reconstruct_matrix_from_csv('unrobustified_sign_correls.csv', n, full = (n==18895) )
time_reconstruction = time.time()-tic
print("Time to reconstruct the matrix from csv file: ", time_reconstruction)


tac = time.time()
sol, nbit , dist = alternating_proj(mat_input, np.diag(np.ones(n)), method = 'classic')
time_alternating = time.time()-tac
print("Time for alternating projection algorithm (full decomposition) : ", time_alternating)
print("Number of iterations : ", nbit)
print("Optimal objective value : ", dist)
tac = time.time()
sol, nbit , dist = alternating_proj(mat_input, np.diag(np.ones(n)))
time_alternating = time.time()-tac
print("Time for alternating projection algorithm (only positive eigenvalues) : ", time_alternating)
print("Number of iterations : ", nbit)
print("Optimal objective value : ", dist)


###############################################################################
## Anderson acceleration of the alternating projection 
def proj_spd(A, delta, method):
    """
    Project A onto the set of symmetric positive semidefinite matrices
    with minimum eigenvalue at least delta.
    """
    if method == "positive only": 
        eigenvalues, eigenvectors = sp.linalg.eigh(A, subset_by_value=(delta, np.inf))
    
    elif method == "classic": 
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    else : 
        print(method)
        raise ValueError("This method is not covered.")
    eigenvalues_new = np.maximum(eigenvalues, delta)
    X = eigenvectors @ np.diag(eigenvalues_new) @ eigenvectors.T
    return (X + X.T) / 2  # Ensure symmetry

def proj_pattern(A, X, pattern):
    """
    Enforce fixed entries from A on X according to pattern.
    If pattern is None or empty, set the diagonal to one.
    """
    X_new = X.copy()
    if pattern is None or pattern.size == 0:
        np.fill_diagonal(X_new, 1)
    else:
        # pattern should be a binary matrix of the same shape as A
        fixed_indices = np.where(pattern)
        X_new[fixed_indices] = A[fixed_indices]
    return X_new

def ap_step(A, Yin, Sin, pattern, delta, method):
    """
    One alternating projections step.
    """
    R = Yin - Sin
    Xout = proj_spd(R, delta, method)
    Sout = Xout - R
    Yout = proj_pattern(A, Xout, pattern)
    return Xout, Yout, Sout

def nearcorr_aa(A, pattern=None, mMax=2, itmax=500, ls_solve='b', delta=0, tol=None, droptol=0, AAstart=1, method ="positive only"):
    """
    Compute the nearest correlation matrix to A via alternating projections with Anderson acceleration.
    
    Parameters:
      A       : (n,n) symmetric matrix.
      pattern : (n,n) binary matrix indicating fixed entries (1 = fixed); if None, force unit diagonal.
      mMax    : History length for acceleration (integer, default 2; 0 means no acceleration).
      itmax   : Maximum number of iterations (default 100).
      ls_solve: Least-squares method; here only 'b' (backslash, i.e. np.linalg.lstsq) is implemented.
      delta   : Minimum eigenvalue threshold (default 0).
      tol     : Convergence tolerance (default: n * eps).
      droptol : Tolerance for dropping stored residual vectors (if > 0, drop columns if condition number exceeds droptol).
      AAstart : Iteration index to start acceleration (default 1).
      
    Returns:
      Y       : Nearest correlation matrix.
      iter    : Number of iterations performed.
    """
    if not np.allclose(A, A.T, atol=1e-10):
        raise ValueError("The input matrix must be symmetric.")
    
    n = A.shape[0]
    if tol is None:
        tol = n * np.finfo(float).eps

    # Initialize storage for Anderson acceleration.
    DG = None  # Will store differences of g-values (each column is a vector)
    DF = None  # Will store differences of f-values
    mAA = 0   # Number of stored residuals

    # Initialize Y and S (S is the dual shift matrix)
    Yin = A.copy()
    Sin = np.zeros_like(A)
    iter_count = 0
    rel_diffXY = np.inf

    # These will hold the previous fval and gval for acceleration updates.
    f_old = None
    g_old = None

    size_n2 = n * n  # number of elements in an n x n matrix

    while rel_diffXY > tol and iter_count < itmax:
        iter_count += 1
        # One alternating projections step.
        Xout, Yout, Sout = ap_step(A, Yin, Sin, pattern, delta, method)
        
        # Vectorize Yin and Sin into one vector x.
        x = np.concatenate([Yin.flatten(), Sin.flatten()])
        gval = np.concatenate([Yout.flatten(), Sout.flatten()])
        fval = gval - x
        
        # Convergence test: relative Frobenius norm of Yout-Xout.
        rel_diffXY = np.linalg.norm(Yout - Xout, 'fro') / np.linalg.norm(Yout, 'fro')
        # Uncomment the following line to monitor convergence:
        # print(f"Iter {iter_count}: rel_diffXY = {rel_diffXY}")
        if rel_diffXY < tol:
            break

        # Without acceleration: if no history or not yet started.
        if mMax == 0 or iter_count < AAstart:
            x_new = gval.copy()
        else:
            # Anderson acceleration step.
            if iter_count > AAstart:
                # Compute differences.
                df = fval - f_old if f_old is not None else fval.copy()
                dg = gval - g_old if g_old is not None else gval.copy()
                # Append new differences.
                if mAA == 0:
                    DG = dg.reshape(-1, 1)
                    DF = df.reshape(-1, 1)
                else:
                    DG = np.column_stack([DG, dg])
                    DF = np.column_stack([DF, df])
                mAA += 1
                # If we have too many stored columns, drop the oldest.
                if mAA > mMax:
                    DG = DG[:, 1:]
                    DF = DF[:, 1:]
                    mAA -= 1
                # Optionally, drop columns if condition number of DF is too high.
                if droptol > 0 and mAA > 1:
                    condDF = np.linalg.cond(DF)
                    while condDF > droptol and mAA > 1:
                        # Drop the oldest column.
                        DG = DG[:, 1:]
                        DF = DF[:, 1:]
                        mAA -= 1
                        condDF = np.linalg.cond(DF)
            # Update stored f_old and g_old.
            f_old = fval.copy()
            g_old = gval.copy()
            
            if mAA == 0:
                x_new = gval.copy()
            else:
                # Using the 'b' option: solve DF * gamma = fval in least-squares sense.
                gamma, residuals, rank, s = np.linalg.lstsq(DF, fval, rcond=None)
                x_new = gval - DG @ gamma

        # Reshape the new iterate x_new into Yin and Sin.
        Yin = x_new[:size_n2].reshape(n, n)
        Sin = x_new[size_n2:].reshape(n, n)
    
    if iter_count >= itmax and rel_diffXY > tol:
        raise RuntimeError(f"Stopped after {itmax} iterations. Try increasing itmax.")
    return Yin, iter_count
toc = time.time()
sol, nbiteraa = nearcorr_aa(mat_input, pattern=None, mMax=2, itmax=500, delta=0, method = "classic")
time_andersonacc = time.time() - toc
print("Time for Anderson acceleration of the alternating scheme (full decomposition): ", time_andersonacc)
print("Number of iterations : ", nbiteraa)
toc = time.time()
sol, nbiteraa = nearcorr_aa(mat_input, pattern=None, mMax=2, itmax=500, delta=0)
time_andersonacc = time.time() - toc
print("Time for Anderson acceleration of the alternating scheme (only positive): ", time_andersonacc)
print("Number of iterations : ", nbiteraa)

