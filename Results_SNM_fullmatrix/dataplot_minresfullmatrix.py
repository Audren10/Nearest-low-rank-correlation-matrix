# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:03:45 2025

@author: audre
"""
import matplotlib.pyplot as plt

# PROVIDED DATA
sizes = [18895]
rank_of_X = 1430

gradN = [1613.3721856664606, 463.20236000533856, 67.47403216549465, 
         27.770833774566267, 22.287919904325133, 19.13264690566383, 
         6.817855474455071, 4.126263049720716, 2.2429294621737954, 
         1.408402428420106, 0.01041239574064618]

dualObj = [4653974.94719585, 7686125.45077585, 8269197.09357737, 
           8316965.97799541, 8317475.19747346, 8318960.9470648, 
           8320907.56690108, 8321120.68633548, 8321199.287685, 
           8321220.26564008, 8321233.57476694, 8321233.57476694]

primalObj = [9156501.055414762, 8788420.223695867, 8741996.449813386, 
             8331700.311489002, 8462027.310475165, 8330419.174938434, 
             8363909.375480823, 8321750.153396923, 8335150.141276191, 
             8321295.8493893985, 8321238.5259637665, 8321238.5259637665]

relDualityGap = [0.06690896, 0.02779342, 0.00088502, 0.0086148, 0.00068821, 
                 0.0025773, 3.78220218e-05, 0.00083757, 4.54160562e-06, 
                 2.9750367e-07, 2.9750367e-07]

iterInn = [1, 2, 3, 3, 2, 2, 2, 2, 2, 1]

execTime = {
    'init': 668.6424484000017,
    'avg step time': 13752.296230700005/len(iterInn),
    'preconditioner': 314.0982156999962,
    'minres time': 1069.7240332999718,
    'eig_time': 10647.839636699995,
    'total time': 13758.308219100007
}

# Creating the plots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

axs[0].plot(primalObj, marker='o', color='blue')
axs[0].set_title("Primal Objective vs Iterations")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("Primal Objective")

axs[1].plot(dualObj, marker='o', color='red')
axs[1].set_title("Dual Objective vs Iterations")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Dual Objective")

axs[2].plot(gradN, marker='o', color='green')
axs[2].set_title("Gradient Norm vs Iterations")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Gradient Norm")

axs[3].plot(relDualityGap, marker='o', color='purple')
axs[3].set_title("Relative Duality Gap vs Iterations")
axs[3].set_xlabel("Iteration")
axs[3].set_ylabel("Relative Duality Gap")

phases = list(execTime.keys())
times = list(execTime.values())
axs[4].bar(phases, times, color='orange')
axs[4].set_title("Execution Time per Phase")
axs[4].set_xlabel("Phase")
axs[4].set_ylabel("Time (s)")
axs[4].tick_params(axis='x', rotation=45)

axs[5].plot(iterInn, marker='o', color='brown')
axs[5].set_title("Inner Iterations vs Outer Iterations")
axs[5].set_xlabel("Outer Iteration")
axs[5].set_ylabel("Number of Inner Iterations")

plt.suptitle(f"Sizes: {sizes}    rank of X: {rank_of_X}", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

import matplotlib.pyplot as plt
# Data for each iteration difference
iteration_diffs = [
    'Iteration 1-0', 'Iteration 2-1', 'Iteration 3-2', 'Iteration 4-3', 'Iteration 5-4',
    'Iteration 6-5', 'Iteration 7-6', 'Iteration 8-7', 'Iteration 9-8', 'Iteration 10-9'
]
means = [
    -22.412280103804225, -32.36995051034251, 82.99652311963364, -80.70272370517516, 12.375777811018315,
    -9.111780751352573, 4.673463792443992, -2.3531730608081762, 1.3527029560106951, -0.5014713826688033
]
variances = [
    21.32571097235132, 57.416559138439624, 765.5198325310563, 518.128104542928, 22.94992514235276,
    5.300953529343628, 1.4412098946026026, 0.3712960627270961, 0.12368352323978657, 0.01647432206532706
]
mins = [
    -50.9439921567448, -72.44433012058704, -85.3680215929482, -188.04746284766134, 1.9567726820597393,
    -22.737712647110357, 0.36592734455306, -5.611891995974801, 0.10031636526285537, -1.2063438708671015
]
maxs = [
    -3.9127919350955205, -2.9202015518400666, 193.5846343099943, -11.697393481488533, 52.17996062496911,
    -0.8267126548476629, 11.417673480728212, -0.17856621655374827, 3.2975312983371765, -0.03683502564446428
]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(iteration_diffs, means, marker='o', linestyle='-', color='blue', label='Mean')
plt.plot(iteration_diffs, variances, marker='s', linestyle='--', color='green', label='Variance')
plt.plot(iteration_diffs, mins, marker='^', linestyle='-.', color='red', label='Minimum')
plt.plot(iteration_diffs, maxs, marker='v', linestyle=':', color='purple', label='Maximum')

plt.title("Evolution of Δy Statistics Across Iterations")
plt.xlabel("Iteration Difference")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
