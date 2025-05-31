# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:28:34 2025

@author: audre
"""

import numpy as np
import matplotlib.pyplot as plt

# PROVIDED DATA
sizes = [18895]
rank_of_X = 1430

gradN = [1613.3721856664606, 463.19690835124254, 123.5378646892163, 
         24.010252080115777, 2.795171093925643, 0.1343966453887533]

dualObj = [4653974.94719585, 7686136.27652022, 8237200.61789177, 
           8315379.28591059, 8321073.60910383, 8321233.02728199, 
           8321233.02728199]

primalObj = [9156501.055414762, 8788416.883270932, 8475193.825769458, 
             8336308.344435112, 8321537.535021861, 8321243.152800136, 
             8321243.152800136]

relDualityGap = [0.06690807, 0.01424052, 0.00125687, 
                 2.78757874e-05, 6.08414123e-07, 6.08414123e-07]

iterInn = [1, 2, 3, 3, 3]

execTime = {
    'init': 549.1004027000017,
    'mean time by iteration': 4052.7540472/len(iterInn),
    'preconditioner': 144.59334849999868,
    'pcg time': 684.5673239000062,
    'eig_time': 2733.752120800007,
    'total time': 4056.1398368000046
}

# For the dual variable, we use dualObj as the sequence (can be adjusted if needed)
y_values = dualObj

# -------------------------------------------------------------------
# First figure: combination of multiple plots

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()  # access via indices 0 to 5

# 1. Primal Objective vs Iterations
iterations_primal = list(range(len(primalObj)))
axs[0].plot(iterations_primal, primalObj, marker='o', color='blue')
axs[0].set_title("Primal Objective vs Iterations")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Primal Objective")

# 2. Dual Objective vs Iterations
iterations_dual = list(range(len(dualObj)))
axs[1].plot(iterations_dual, dualObj, marker='o', color='red')
axs[1].set_title("Dual Objective vs Iterations")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Dual Objective")

# 3. Gradient Norm vs Iterations
iterations_grad = list(range(len(gradN)))
axs[2].plot(iterations_grad, gradN, marker='o', color='green')
axs[2].set_title("Gradient Norm vs Iterations")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Gradient Norm")

# 4. Relative Duality Gap vs Iterations
iterations_gap = list(range(len(relDualityGap)))
axs[3].plot(iterations_gap, relDualityGap, marker='o', color='purple')
axs[3].set_title("Relative Duality Gap vs Iterations")
axs[3].set_xlabel("Iteration")
axs[3].set_ylabel("Relative Duality Gap")

# 5. Execution time per phase (bar chart)
phases = list(execTime.keys())
times = list(execTime.values())
axs[4].bar(phases, times, color='orange')
axs[4].set_title("Execution Time per Phase")
axs[4].set_xlabel("Phase")
axs[4].set_ylabel("Time (s)")
axs[4].tick_params(axis='x', rotation=45)

# 6. Number of inner iterations vs outer iterations
iterations_inn = list(range(len(iterInn)))
axs[5].plot(iterations_inn, iterInn, marker='o', color='brown')
axs[5].set_title("Inner Iterations vs Outer Iterations")
axs[5].set_xlabel("Outer Iteration")
axs[5].set_ylabel("Number of Inner Iterations")

plt.suptitle(f"Sizes: {sizes}    rank of X: {rank_of_X}", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -------------------------------------------------------------------
# Second figure: evolution of the norms of differences between y_i and y_{i-1}
# Based on your data, the norms are as follows:
delta_y_norm = [3145.5126479970254, 2014.918789644392, 1147.3401731035933, 
                490.4221095521831, 122.7205288658941]

mean_diff = np.mean(delta_y_norm)
var_diff = np.var(delta_y_norm)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(delta_y_norm)+1), delta_y_norm, marker='o', linestyle='-', color='magenta')
plt.title("Evolution of Norms of Differences (y_i - y_{i-1})")
plt.xlabel("Iteration Difference (i -> i+1)")
plt.ylabel("Norm of (y_i - y_{i-1})")
plt.grid(True)

# Display mean and variance on the graph
plt.text(0.7, 0.95, f"Mean = {mean_diff:.2f}\nVariance = {var_diff:.2f}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data for each iteration difference
iteration_diffs = ['Iteration 1-0', 'Iteration 2-1', 'Iteration 3-2', 'Iteration 4-3', 'Iteration 5-4']
means = [-22.41, -14.06, -7.25, -2.07, -0.25]
variances = [21.33, 17.23, 17.05, 8.46, 0.73]
mins = [-50.94, -38.27, -43.44, -36.71, -16.55]
maxs = [-3.91, -0.27, 1.96, 1.25, 0.33]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(iteration_diffs, means, marker='o', linestyle='-', color='blue', label='Mean')
plt.plot(iteration_diffs, variances, marker='s', linestyle='--', color='green', label='Variance')
plt.plot(iteration_diffs, mins, marker='^', linestyle='-.', color='red', label='Minimum')
plt.plot(iteration_diffs, maxs, marker='v', linestyle=':', color='purple', label='Maximum')

plt.title("Evolution of Δy Statistics Across Iterations")
plt.xlabel("Iteration Difference")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

