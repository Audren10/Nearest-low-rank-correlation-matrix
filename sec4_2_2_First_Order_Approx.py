import numpy as np
import time 
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# Load the dual vectors into memory
###############################################################################

def read_vectors_from_file(filename):
    vectors = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Search for lines indicating vector entries (e.g., "Vector for entry X:")
            if lines[i].startswith('Vector for entry'):
                # Extract the index of the entry
                entry_index = int(lines[i].split()[3].strip(':'))
                i += 1  # Move to the next line with values
                
                # Read the values of the vector
                vector_values = []
                while i < len(lines) and lines[i].strip() != '':
                    vector_values.append(float(lines[i].strip()))
                    i += 1
                
                # Convert list of values into a NumPy array
                vectors[entry_index] = np.array(vector_values)
            i += 1

    return vectors

# Example usage
vectors = read_vectors_from_file('output_vectors.txt')

# Display the first vector (for example)
print(vectors[0])

def first_order_perturbation_vectorized(lam, U, delty):
    """
    Compute first-order perturbation corrections to the eigenvalues and eigenvectors
    of a Hermitian matrix M perturbed by Diag(y).

    Parameters:
      lam : np.ndarray
          Array of initial eigenvalues, sorted in descending order.
      U : np.ndarray
          Matrix of initial eigenvectors.
      delty : np.ndarray
          Perturbation vector of size (n,), used in Diag(y).
          
    Returns:
      lam_pert : np.ndarray
          Approximated eigenvalues of M + Diag(y).
      U_pert : np.ndarray
          Matrix whose columns are the approximated eigenvectors.
    """
    # Compute correction matrix A = U.T @ Diag(y) @ U efficiently
    A = delty[:,None] * U
    print("Element-wise product finished")
    A = np.dot(U.T, A)
    print("Computation of A finished")
    # Compute weighting matrix 
    lam = lam.flatten()
    diffs = lam[:, None] - lam[None, :]  # (n, n) matrix    
    np.fill_diagonal(diffs, np.inf)  # avoid division by zero on the diagonal 
    # --- Correction to the eigenvalues ---
    lam = lam + np.diag(A)
    # --- Correction to the eigenvectors --- 
    A = A.transpose() / diffs
    # Compute corrections to the eigenvectors
    U = U + np.dot(U, A)
    print("Correction to eigenvectors finished")
    # Normalize each eigenvector (each column)
    U = U / np.linalg.norm(U, axis=0, keepdims=True)
    return lam, U

def reconstruct_matrix_from_csv(file_path, matrix_size, full=False):
    """
    Reconstructs a full symmetric correlation matrix from a CSV file containing the lower-left part.

    Parameters:
        file_path (str): Path to the CSV file containing the lower-left part of the matrix.
        matrix_size (int): Size of the full square matrix (number of rows/columns).
        full (bool): If True, assume the full matrix is stored row-wise.

    Returns:
        np.ndarray: The reconstructed symmetric correlation matrix.
    """
    if full:
        with open(file_path, 'r') as f:
            data = f.read().strip()
        values = np.array(list(map(float, data.split(',')))) 
    else:
        # Compute number of values to read (lower triangular part without diagonal)
        num_values = matrix_size * (matrix_size - 1) // 2
        
        # Read values from the CSV file into a 1D NumPy array
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
    
    # Initialize empty square matrix
    matrix = np.zeros((matrix_size, matrix_size))

    # Fill lower triangular part
    index = 0
    for j in range(0, matrix_size - 1):
        for i in range(j + 1, matrix_size):
            matrix[i, j] = values[index]
            index += 1

    # Symmetrize the matrix
    matrix += matrix.T

    # Set diagonal to 1 (correlation matrix)
    np.fill_diagonal(matrix, 1.0)

    return matrix

# Main block
if __name__ == "__main__":
    tic = time.time()
    M = reconstruct_matrix_from_csv('unrobustified_sign_correls.csv', 18895, full=True)
    toc = time.time() - tic
    print("Time to reconstruct the matrix:", toc)

    # Compute eigenvalues and eigenvectors of M + diag(vectors[4])
    tac = time.time()
    lam, U = np.linalg.eigh(M + np.diag(vectors[4]))
    tuc = time.time() - tac
    print("Time to compute eigenvalue decomposition:", tuc)

    # Reverse order to get descending eigenvalues
    lam = lam[::-1]
    U = U[:, ::-1]

    # Construct perturbation vector
    y = vectors[5] - vectors[4]

    # Apply first-order perturbation theory
    tictic = time.time()
    lam_pert, U_pert = first_order_perturbation_vectorized(lam, U, y)
    toctoc = time.time() - tictic
    print("Time to compute first-order approximation:", toctoc)

    # Compute exact eigenvalues and eigenvectors of perturbed matrix
    tactac = time.time()
    lam_exact, U_exact = np.linalg.eigh(M + np.diag(vectors[5]))
    tuctuc = time.time() - tactac
    print("Time to compute eigenvalue decomposition (exact):", tuctuc)
    lam_exact = lam_exact[::-1]
    U_exact = U_exact[:, ::-1]
    
    ### analysis of eigenvalues 
    # Compute absolute eigenvalue errors
    eigenvalue_errors = np.abs(lam_exact - lam_pert)
    mean_error = np.mean(eigenvalue_errors)
    max_error = np.max(eigenvalue_errors)

    print(f"\nMean error: {mean_error:.5f}")
    print(f"Max error: {max_error:.5f}")

    # Plot: Boxplot + Histogram
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 6), sharex=True)

    # Boxplot on top
    sns.boxplot(x=eigenvalue_errors, ax=ax[0], orient='h', color='lightblue')
    ax[0].set(xlabel='')

    # Histogram below
    sns.histplot(eigenvalue_errors, bins=60, kde=True, ax=ax[1], color='blue')
    ax[1].set_xlabel("Eigenvalue errors")
    ax[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Compute relative errors
    rel_error = np.abs(eigenvalue_errors / lam_exact)

    # Statistics
    mean_error = np.mean(rel_error)
    std_dev = np.std(rel_error)
    max_error = np.max(rel_error)
    outliers = rel_error[(rel_error > 15 * std_dev)]

    # Plot relative error boxplot
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.boxplot(
        x=rel_error, ax=ax, color='lightblue', width=0.5, showcaps=True,
        boxprops={'edgecolor': 'black', 'facecolor': 'cyan', 'linewidth': 2},
        whiskerprops={'color': 'black', 'linewidth': 2}, 
        medianprops={'color': 'red', 'linewidth': 2}, 
        capprops={'color': 'black', 'linewidth': 2}
    )

    # Highlight ±1 std deviation band
    ax.axvspan(mean_error - std_dev, mean_error + std_dev, color='orange', alpha=0.3, label="±1σ")

    # Lines for mean and max
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_error:.2e}')
    ax.axvline(max_error, color='purple', linestyle='-', linewidth=2, label=f'Std dev = {std_dev:.2e}')

    # Optional: annotate top outliers
    for outlier in outliers[:5]:
        ax.text(outlier, 0.02, f'{outlier:.2e}', rotation=45, color='black')

    ax.legend()
    ax.set_title("Relative Eigenvalue Errors")
    plt.tight_layout()
    plt.show()
    
    from sklearn.linear_model import LinearRegression

    # Make sure dy is available: dy = Delta y vector of perturbations
    dy_abs = y.reshape(-1, 1)  # reshape for sklearn
    
    # Absolute error vs |dy|
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Plot 1: Absolute error vs perturbation ---
    ax[0].scatter(dy_abs, eigenvalue_errors, alpha=0.6, label='Error')
    # Linear fit
    reg_abs = LinearRegression().fit(dy_abs, eigenvalue_errors)
    ax[0].plot(dy_abs, reg_abs.predict(dy_abs), color='red', label='Linear fit')
    ax[0].set_xlabel('|Perturbation Δy|')
    ax[0].set_ylabel('Absolute eigenvalue error')
    ax[0].set_title('Absolute Error vs Perturbation')
    ax[0].legend()
    
    # --- Plot 2: Relative error vs perturbation ---
    rel_error = np.abs(eigenvalue_errors / lam_exact)
    ax[1].scatter(dy_abs, rel_error, alpha=0.6, label='Relative error')
    # Linear fit
    reg_rel = LinearRegression().fit(dy_abs, rel_error)
    ax[1].plot(dy_abs, reg_rel.predict(dy_abs), color='red', label='Linear fit')
    ax[1].set_xlabel('|Perturbation Δy|')
    ax[1].set_ylabel('Relative eigenvalue error')
    ax[1].set_title('Relative Error vs Perturbation')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()

    ### analysis of eigenvectors 
    print('Max error between eigenvectors: ', np.max(np.abs(U_exact - U_pert)))
    print('Frobenius norm of the difference : ', np.linalg.norm(U_pert-U_exact, 'fro')) 
    print("Dot product matrix (should be close to identity):", np.linalg.norm(U_pert.T @ U_pert- np.eye(18895), 'fro'))


    # Compute angles (in radians) between corresponding eigenvectors (column-wise)
    angles_rad = np.arccos(np.clip(np.abs(np.sum(U_exact * U_pert, axis=0)), 0, 1))
    angles_deg = np.degrees(angles_rad)
    # Plot: Boxplot + Histogram
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 6), sharex=True)
    # Top: Boxplot of angles
    sns.boxplot(x=angles_deg, ax=ax[0], orient='h', color='lightblue')
    ax[0].set(xlabel='')
    ax[0].set_title("Distribution of angles between exact and perturbed eigenvectors (degrees)")
    
    # Bottom: Histogram with density estimate
    sns.histplot(angles_deg, bins=60, kde=True, ax=ax[1], color='blue')
    ax[1].set_xlabel("Angle (degrees)")
    ax[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    # Scatter plot: Angle vs Perturbation y
    plt.figure(figsize=(10, 6))
    plt.scatter(y, angles_deg, alpha=0.6, edgecolor='k')
    plt.xlabel("Perturbation $y_i$")
    plt.ylabel("Angle between eigenvectors (degrees)")
    plt.title("Angle vs Perturbation applied to diagonal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()