###############################################################################
# Visualization : set of correlation matrices in 3D                          ##
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

# Define the function to check positive semidefiniteness
def is_positive_semidefinite(x, y, z):
    matrix = np.array([[1, x, y],
                       [x, 1, z],
                       [y, z, 1]])
    eigenvalues = np.linalg.eigvalsh(matrix)  # Get eigenvalues
    return np.all(eigenvalues >= 0)  # Check if all are non-negative

# Define the range of x, y, z values
grid_points = 100  # Number of points per axis
x_vals = np.linspace(-1, 1, grid_points)
y_vals = np.linspace(-1, 1, grid_points)
z_vals = np.linspace(-1, 1, grid_points)

# Collect valid (x, y, z) points
valid_points = []
for x in x_vals:
    for y in y_vals:
        for z in z_vals:
            if is_positive_semidefinite(x, y, z):
                valid_points.append((x, y, z))

valid_points = np.array(valid_points)

# Plot the points in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], s=1, alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Elliptote : Set of Correlation Matrices')

plt.show()
###############################################################################

import numpy as np
import matplotlib.pyplot as plt


# Define the grid of x, y values
grid_points = 100  # Number of points per axis
x = np.linspace(-1, 1, grid_points)
y = np.linspace(-1, 1, grid_points)
x, y = np.meshgrid(x, y)

# Compute z values based on the determinant equation
# Solve for z where 1 - y^2 - z^2 - x^2 + 2xyz = 0
z_squared = 1 - y**2 - x**2 + 2 * x * y

# Mask invalid points (where z_squared < 0)
z_squared[z_squared < 0] = np.nan
z = np.sqrt(z_squared)  # Positive branch
z_neg = -np.sqrt(z_squared)  # Negative branch

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot positive branch
ax.plot_surface(x, y, z, alpha=0.7, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# Plot negative branch
ax.plot_surface(x, y, z_neg, alpha=0.7, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

# Set plot labels and limits
ax.set_title("Surface defined by $1 - y^2 - z^2 - x^2 + 2xyz = 0$")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()

import sympy as sp

# Define symbolic variables
x, y, z = sp.symbols('x y z')

# Define the matrix
A = sp.Matrix([[1, x, y],
               [x, 1, z],
               [y, z, 1]])

# Compute the eigenvalues
eigenvalues = A.eigenvals()
print("Eigenvalues (symbolic):")
print(eigenvalues)

