# -*- coding: utf-8 -*-
"""
Created on Fri May 16 04:27:58 2025

@author: audre
"""
## Sanity check for the gradient of the obejctive function
## Scaled-Subspace Restriction Method 
import numpy as np
import matplotlib.pyplot as plt
import time

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

# Download the full data
n = 100
M = reconstruct_matrix_from_csv('unrobustified_sign_correls.csv', n, full=(n==18895))
# Spectral decomposition
eigvals, eigvecs = np.linalg.eigh(M)
# Sort in descending order
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]

# Keep the k largest eigenvalues and associated eigenvectors
k = 450
eigvals = eigvals[:k]
U = eigvecs[:, :k]

# Initialiser Z
Z = np.eye(k)

# Fonction f(Z)
def f(Z):
    UZUt = U @ Z @ U.T
    D = np.diag(1.0 / np.sqrt(np.diag(UZUt)))
    DUZUtD = D @ UZUt @ D
    return np.linalg.norm(M - DUZUtD, 'fro')**2

def grad_f(Z):
    UZUt = U @ Z @ U.T
    D = np.diag(1.0 / np.sqrt(np.diag(UZUt)))
    DUZUtD = D @ UZUt @ D
    M_minus_DUZUtD = M - DUZUtD
    D3 = D @ D @ D
    term1 = U.T @ np.diag(np.diag(UZUt @ D @ M_minus_DUZUtD @ D3)) @ U
    term2 = 2 * (U.T @ D @ M_minus_DUZUtD @ D @ U)
    term3 = U.T @ np.diag(np.diag(M_minus_DUZUtD @ D @ UZUt @ D3)) @ U
    return term1 - term2 + term3

# Perturbation Z'
Z_prime = np.random.randn(k, k)
Z_prime = (Z_prime + Z_prime.T) / 2  # Symmetrize the perturbation

t1 = time.time()

# Compute the gradient at point Z
G = grad_f(Z)
f_0 = f(Z)

# Log-log plot of the error
ts = np.logspace(-8, 0, 100)
errors = []

for t in ts:
    f_t = f(Z + t * Z_prime)  # Evaluate function at perturbed point
    lin_approx = f_0 + t * np.sum(G * Z_prime)  # First-order Taylor approximation
    errors.append(abs(f_t - lin_approx))  # Absolute difference

print(time.time() - t1)

# Display log-log plot
plt.figure(figsize=(7, 5))
plt.loglog(ts, errors, label=r"$f(Z + tZ') - f(Z) - t \langle \nabla f(Z), Z' \rangle$")
plt.loglog(ts, [ts[i]**2 * errors[5] / (ts[5]**2) for i in range(len(ts))], '--', label=r"$t^2$ slope")
plt.xlabel("log(t)")
plt.ylabel("log(error)")
plt.title("Gradient check for $f(Z)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

