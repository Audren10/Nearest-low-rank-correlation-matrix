# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 04:10:27 2025

@author: audre
"""
import matplotlib.pyplot as plt
import numpy as np
# PROVIDED DATA
results = {
    
    "N=5000": {
        "CG": {
            "method": "CG with preconditioning",
            "iterations": [1, 2, 3, 4, 5, 6, 7, 8],
            "dual_obj": [
                326726.35611661, 535712.99124199, 564371.7984082, 571530.64754154,
                571625.52078107, 571630.52900741, 571630.9312831, 571634.36328658
            ],
            "primal_obj": [
                634119.3900124802, 595430.7470529167, 622667.134423907, 573710.2953195022,
                571912.9544752403, 572794.09905813, 571642.3937722368, 571634.4272409343
            ],
            "gap": [
                0.05279409, 0.04910984, 0.00190322, 0.00025135, 0.00101673, 1.00260181e-05,
                0.00028954, 5.59398577e-08
            ],
            "grad_norm": [
                429.8083051520422, 117.44449207683309, 34.91661893822266, 3.726480566321664,
                1.2436365114682275, 0.8911364337040999, 0.9022378662656602, 0.06907310922173407
            ],
            "time": [
                13.624155800091103, 29.477441600058228, 47.7556170001626, 99.63597900001332,
                126.98242280003615, 153.9525077000726, 181.22303800005466, 231.9625507001765
            ]
        },
        "MINRES": {
            "method": "MINRES with preconditioning",
            "iterations": [1, 2, 3, 4, 5, 6, 7, 8],
            "dual_obj": [
                326800.54667256, 535798.24024912, 564453.92892501, 571589.74164359,
                571665.62844293, 571681.04937911, 571681.25492372, 571684.76341299
            ],
            "primal_obj": [
                634230.123478201, 595520.125838013, 622790.467026581, 573740.1437099684,
                571964.8201567644, 572839.02134445, 571661.4471588129, 571653.5510922225
            ],
            "gap": [
                0.05260089, 0.04876889, 0.00194491, 0.00024474, 0.00100294, 1.01720675e-05,
                0.00028115, 5.71789217e-08
            ],
            "grad_norm": [
                429.6166228334359, 117.27664004581818, 34.85967818281358, 3.7102593802630856,
                1.2428360610988354, 0.8884383038492139, 0.900881287616107, 0.0689730220939072
            ],
            "time": [
                13.664889804023908, 29.498831458606373, 47.80364058948194, 99.70389411802244,
                127.0402717257488, 154.03416268519498, 181.3300677022015, 232.02128432050875
            ]
        }
    },
    "N=10000": {
        "CG": {
        "method": "CG with preconditioning",
        "iterations": range(0,6),
        "inner_iter": [0,1,2,2,3,3], 
        "dual_obj": [
            1100505.1811175, 1803521.31584686, 1918485.8497419, 1931854.47160665, 
            1932596.39, 1932610.38
        ],
        "primal_obj": [
            2175653.5508164074, 2046780.5657266546, 1958173.8146900046, 1933930.0736, 
            1932625.3253, 1932610.79
        ],
        "gap": [
            0, 0.06317925, 0.01023767, 0.00053692, 7.48e-6, 1.04e-7
        ],
        "grad_norm": [
            783.4228303851726, 217.48252529484546, 53.679161148980164, 9.0114, 
           
            0.8945, 0.032
        ],
        "time": [
            108.62177970004268, 206.59435350005515, 312.5943849000614, 406.753, 
            507.103, 606.3674
        ]
    }, 
        "MINRES": {
        "method": "MINRES",
        "iterations": range(0,8),
        "inner_iter": [0,1,2,3,3,3,3,2], 
        "dual_obj": [1100505.1811175, 1803517.9689237, 1914295.40890661, 1932276.06376862, 1932594.68740529, 1932605.92303629, 1932607.16, 1932610.20],
        "primal_obj": [2175653.5508164074, 2046782.1078958448, 2070156.1608436767, 1937150.2999500318, 1934816.282218641, 1932632.0556982278, 1934689.69, 1932611.168076384],
        "gap": [0, 0.06318055, 0.03911723, 0.00125968, 0.00057444, 6.76094339e-06, 0.0005385, 2.48393942e-07],
        "grad_norm": [783.4228303851726, 217.48627345103048, 48.82081325966155, 6.2758029471703205, 1.5057304566874739, 0.948527633522199, 0.82, 0.1963],
        "time": [78.97337620006874, 175.61667870008387, 280.37387600005604, 587.5930302999914, 764.1143018000294, 938.9334187998902, 1140.63, 1280.8183218000922]
    }
    },
    "N=25": {
        "CG": {
            "method": "CG with preconditioning",
            "iterations": range(0, 5),
            "inner_iter": [0, 2, 2, 2, 2],
            "dual_obj": [6.43882383, 9.26509665, 9.30330965, 9.30339073, 9.3339],
            "primal_obj": [11.561542341212297, 9.358985148420805, 9.305253328509629, 9.303395095595898, 9.30339],
            "gap": [0, 0.00478435, 9.91241019e-05, 1.60867848e-06, 2.19e-08],
            "grad_norm": [1.8445689927078455, 0.18130804292296074, 0.007790292872507179,4.97e-05, 3.12e-7],
            "time": [0.2780034001916647, 0.3205411001108587, 0.32466610008850694, 0.32698099979758263, 0.331619]
        },
        "MINRES": {
        "method": "MINRES",
        "iterations": range(0,17),
        "inner_iter": [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        "dual_obj": [6.43882383, 7.89255999, 9.25054674, 9.2569, 9.3029, 9.3033, 9.3033, 9.3033, 9.3033, 9.3033, 9.3033, 9.3033, 9.3033, 9.3033, 9.3033, 9.3033, 9.30339073],
        "primal_obj": [11.561542341212297, 13.365506199988848 , 9.3808, 9.9093, 9.3101, 9.3044, 9.3047, 9.3050, 9.3049, 9.3057, 9.3050, 9.3036, 9.3034, 9.3037, 9.3035, 9.303395095595898,9.3034],
        "gap": [0, 0.24588597,0.00663648, 0.0323, 0.0003, 5.21e-5, 7.13e-5, 8.44e-5, 7.91e-5, 0.0001, 8.73e-5, 1.11e-5, 4.68e-6, 1.54e-5, 6.79e-6, 2.03e-5, 2.2257149e-07],
        "grad_norm": [1.8445689927078455 , 1.1300212207772726, 0.2366, 0.2172, 0.0216, 0.0011, 0.0012, 0.0014, 0.0017, 0.0019, 0.0022, 0.0001, 0.0001,0.0002 , 0.0002, 0.0002, 2.17e-6],
        "time": [0.10454189986921847 , 0.12346279993653297, 0.1272, 0.1308, 0.1326, 0.1355, 0.1373, 0.1386, 0.1399, 0.1417, 0.1430, 0.1446, 0.1459, 0.1472, 0.1579, 0.1629, 0.1631]
    }
    },
    "N=100": {
        "CG": {
        "method": "CG with preconditioning",
        "iterations": range(0,5),
        "inner_iter": [0,2,2,2,2], 
        "dual_obj": [117.96003402, 167.46894567, 170.64292409, 170.73086262, 170.73],
        "primal_obj": [201.02663136977714, 173.0285922913185, 170.87134992437478, 170.73242413, 170.73],
        "gap": [0, 0.0162802, 0.00066691, 9.15374395e-06, 2.72e-7],
        "grad_norm": [7.1981230930278, 1.3417765906253234, 0.18852136816948994, 0.0075, 4e-5],
        "time": [0.16798679996281862, 0.19915000000037253, 0.2538489999715239, 0.261134, 0.26613]
        }, 
        "MINRES": {
            "method": "MINRES with preconditioning",
            "iterations": range(0,7),
            "inner_iter":[0,2,2,2,2,2,2], 
            "dual_obj":[117.96003402, 159.81661416, 169.73623485, 169.8417892, 170.71727457, 170.73102496, 170.73],
            "primal_obj": [201.02663136977714, 210.87442844020032, 171.78915564517078, 181.5216827977883, 170.77049900501845, 170.77,  170.73133269502264],
            "gap": [0, 0.13736628, 0.00599349, 0.03314729, 0.00015541, 0.00011713, 8.98588474e-07],
            "grad_norm": [7.1981230930278, 3.0372456177112626, 0.9889235326629209, 0.9008205558448396, 0.11966945633370564, 0.0047 ,  0.00026549352517355016], 
            "time": [0.10073420009575784, 0.11031130002811551, 0.11742340005002916, 0.13285279995761812, 0.1395767000503838, 0.1459, 0.15849580010399222]
        },
    },
    "N=500": {
        "CG": {
            "method": "CG with preconditioning",
            "iterations": range(0,5),
            "inner_iter": [0,2,2,2,2], 
            "dual_obj": [1986.19337744, 3050.08669028, 3111.43107518, 3113.43, 3113.10],
            "primal_obj": [3471.580993580078, 3145.037964371081, 3113.9520067913213, 3113.16, 3113.13],
            "gap": [0, 0.0153243, 0.00040488, 9.5816899e-06, 5.27e-7],
            "grad_norm": [32.962311226539995, 6.232023729616861, 0.7401742809477359, 0.06421, 0.001644],
            "time": [0.32959310011938214, 0.46491310000419617, 0.6444123000837862, 0.7544, 0.8360]
        },
        "MINRES": {
        "method": "MINRES",
        "iterations": range(0,7),
        "inner_iter": [0,2,2,2,2,2,2], 
        "dual_obj": [1986.19337744, 2922.87821663, 3055.85439408, 3072.7448247, 3110.73492175, 3113.12749445, 3113.12],
        "primal_obj": [3471.580993580078, 3917.5889447683207, 3144.3163489702915, 3483.561614035583, 3114.621832226548,3117.45, 3113.1309688827073],
        "gap": [0, 0.14539436, 0.01426536, 0.06265023, 0.00062427, 0.000695, 5.57939693e-07],
        "grad_norm": [32.962311226539995, 9.80535736473268, 6.578068391421634, 4.821326607447961, 1.2445656720713534, 0.0636, 0.00545150272786945],
        "time": [0.258971800096333, 0.3614837999921292, 0.49956790008582175, 0.5773193000350147, 0.676440600072965, 0.8455240000039339, 0.8473]
    },
    }
}

sizes =  ["N=25", "N=100", "N=500", "N=10000"] #, "N=2000", "N=5000", "N=1000"]
for s in sizes : 
    plt.figure(figsize=(12, 6))
    
    # Tracer l'évolution des objectifs primal et dual pour CG et MINRES
    plt.subplot(2, 2, 1)
    plt.plot(results[s]["CG"]["iterations"], results[s]["CG"]["dual_obj"], label="CG Dual Objective", marker="o")
    plt.plot(results[s]["CG"]["iterations"], results[s]["CG"]["primal_obj"], label="CG Primal Objective", marker="s")
    plt.plot(results[s]["MINRES"]["iterations"], results[s]["MINRES"]["dual_obj"], label="MINRES Dual Objective", marker="o", linestyle="--")
    plt.plot(results[s]["MINRES"]["iterations"], results[s]["MINRES"]["primal_obj"], label="MINRES Primal Objective", marker="s", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Value")
    plt.title(f"Evolution of Primal and Dual Objectives for {s}")
    plt.legend()
    plt.grid()
    
    # Tracer l'évolution du gap de dualité pour CG et MINRES
    plt.subplot(2, 2, 2)
    plt.plot(results[s]["CG"]["iterations"], results[s]["CG"]["gap"], label="CG Duality Gap", marker="x")
    plt.plot(results[s]["MINRES"]["iterations"], results[s]["MINRES"]["gap"], label="MINRES Duality Gap", marker="x", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Gap")
    plt.title(f"Evolution of Duality Gap for {s}")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    
    # Tracer l'évolution de la norme du gradient pour CG et MINRES
    plt.subplot(2, 2, 3)
    plt.plot(results[s]["CG"]["iterations"], results[s]["CG"]["grad_norm"], label="CG Gradient Norm", marker="^")
    plt.plot(results[s]["MINRES"]["iterations"], results[s]["MINRES"]["grad_norm"], label="MINRES Gradient Norm", marker="^", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Norm of Gradient")
    plt.title(f"Evolution of Gradient Norm for {s}")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    
    # Tracer l'évolution du temps de calcul pour CG et MINRES
    plt.subplot(2, 2, 4)
    plt.plot(results[s]["CG"]["iterations"], results[s]["CG"]["time"], label="CG Time", marker="d")
    plt.plot(results[s]["MINRES"]["iterations"], results[s]["MINRES"]["time"], label="MINRES Time", marker="d", linestyle="--")
    plt.xlabel("Iterations")
    plt.ylabel("Time (s)")
    plt.title(f"Computation Time per Iteration for {s}")
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()


    # Données d'exemple
    iterations_cg = list(results[s]["CG"]["iterations"])  # Convertir en liste
    iterations_minres = list(results[s]["MINRES"]["iterations"])  # Convertir en liste
    
    inner_iter_cg = results[s]["CG"]["inner_iter"]
    inner_iter_minres = results[s]["MINRES"]["inner_iter"]
    
    # Trouver la longueur maximale des deux séries
    max_length = max(len(inner_iter_cg), len(inner_iter_minres))
    
    # Si les longueurs des séries sont différentes, ajouter des valeurs NaN pour combler les manques
    if len(inner_iter_cg) < max_length:
        inner_iter_cg += [np.nan] * (max_length - len(inner_iter_cg))
        iterations_cg += [np.nan] * (max_length - len(iterations_cg))
    
    if len(inner_iter_minres) < max_length:
        inner_iter_minres += [np.nan] * (max_length - len(inner_iter_minres))
        iterations_minres += [np.nan] * (max_length - len(iterations_minres))
    
    # Créer une figure avec deux sous-graphes côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tracer les barres pour CG dans le premier sous-graphe
    axes[0].bar(np.arange(len(inner_iter_cg)), inner_iter_cg, color="b", label="CG Inner Iterations")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Inner Iterations")
    axes[0].set_title(f"CG Inner Iterations for {s}")
    axes[0].set_xticks(np.arange(len(inner_iter_cg)))
    axes[0].set_xticklabels(iterations_cg, rotation=90)
    axes[0].grid(True)
    axes[0].legend()
    
    # Tracer les barres pour MINRES dans le deuxième sous-graphe
    axes[1].bar(np.arange(len(inner_iter_minres)), inner_iter_minres, color="r", label="MINRES Inner Iterations")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Inner Iterations")
    axes[1].set_title(f"MINRES Inner Iterations for {s}")
    axes[1].set_xticks(np.arange(len(inner_iter_minres)))
    axes[1].set_xticklabels(iterations_minres, rotation=90)
    axes[1].grid(True)
    axes[1].legend()
    
    # Ajuster l'espacement entre les graphiques
    plt.tight_layout()
    
    # Afficher le graphe
    plt.show()
    
