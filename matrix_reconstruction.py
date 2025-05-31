# -*- coding: utf-8 -*-
"""
Created on Sat May 31 08:34:30 2025

@author: audre
"""
import numpy as np 

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

if __name__ == "__main__":
    #Load input matrix of the size of your choice
    n = 100
    mat_input = reconstruct_matrix_from_csv('unrobustified_sign_correls.csv', n, full=False) 