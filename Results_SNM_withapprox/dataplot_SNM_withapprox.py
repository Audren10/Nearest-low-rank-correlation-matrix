# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:03:31 2025

Updated: May 30, 2025 by Audren
"""

import matplotlib.pyplot as plt
import numpy as np

# Iteration range
iterations = np.arange(1, 15)

# Gradient norm (14 values, so exclude final step)
gradient_norm = [
    1613.3721856664606, 463.1969083512425, 123.53786468921632,
    24.010252080115713, 2.795171093925713, 4.332147541988293,
    7.37516723701589, 10.815048845583465, 14.040649148677234,
    17.417345658141596, 21.660326247292396, 26.76610812129955,
    32.6096068654875
]

# Objective values
dual_obj = [
    4653974.94719585, 7686136.27652022, 8237200.61789177, 8315379.28591059,
    8321073.60910383, 8321356.67113527, 8322192.41098202, 8324426.65532601,
    8328766.16177061, 8335360.38365231, 8344310.91944607, 8355814.76578455,
    8369551.04936453, 8385367.28579512,
]

primal_obj = [
    9156501.055414762, 8788416.883270932, 8475193.825769458,
    8336308.344435113, 8321537.535021861, 8329905.259449233,
    8347489.253131331, 8399794.74338693, 8491921.517228182,
    8599726.834183969, 8735753.957218204, 8890563.8256657,
    9063117.205816621, 9251356.649666838
]

rel_duality_gap = [
    0.06690807, 0.01424052, 0.00125687, 2.78757874e-05, 0.00051339,
    0.00151754, 0.00450652, 0.00969968, 0.01561057, 0.02291812,
    0.03170933, 0.04193129, 0.0536014, 0.06672125
]

# Rank of X (number of strictly positive eigenvalues)
rank_X = [
    9718, 4427, 2287, 1571, 1437, 1430, 1444, 1467,
    1499, 1536, 1577, 1629, 1678, 1714
]

# Plotting
plt.figure(figsize=(16, 12))

# Plot primal and dual objective values
plt.subplot(2, 2, 1)
plt.plot(iterations, primal_obj, marker='o', label='Primal Objective')
plt.plot(iterations, dual_obj, marker='s', label='Dual Objective')
plt.title("Primal and Dual Objective Values")
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.legend()
plt.grid(True)

# Plot relative duality gap (log scale)
plt.subplot(2, 2, 2)
plt.plot(iterations, rel_duality_gap, marker='o', color='black')
plt.title("Relative Duality Gap")
plt.xlabel("Iteration")
plt.ylabel("Relative Gap")
plt.yscale('log')
plt.grid(True)

# Plot gradient norm (log scale)
plt.subplot(2, 2, 3)
plt.plot(iterations[:-1], gradient_norm, marker='o', color='green')
plt.title("Gradient Norm")
plt.xlabel("Iteration")
plt.ylabel("Norm")
plt.yscale('log')
plt.grid(True)

# Plot rank of X
plt.subplot(2, 2, 4)
plt.plot(iterations, rank_X, marker='o', linestyle='-', color='blue')
plt.title("Rank of X (Positive Eigenvalues)")
plt.xlabel("Iteration")
plt.ylabel("Rank")
plt.grid(True)

plt.tight_layout()
plt.show()
