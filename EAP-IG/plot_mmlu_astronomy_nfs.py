import os
import sys
import ast
import json
import bisect
import numpy as np
import random
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# # ioi
# topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
# new_topns = [int((topns[k]+topns[k-1])/2) for k in range(1, len(topns))]

# all_best_results = [0.19499148, 0.31752195, 0.5808903,  0.77888358, 0.86223119, 0.92848419,                          
#  0.94087017, 0.98918548, 0.98967071, 1.02308339]    
                           
# all_vanilla_results = [0.07630908, 0.09988146, 0.26262252, 0.45394956, 0.43870284, 0.53645372,
#  0.5802243,  0.63649655, 0.6393051,  0.65073048]
                               
# all_avg_results = [0.03121914, 0.13172537, 0.3471377,  0.40624448, 0.51670851, 0.59518838,
#  0.62821324, 0.63241261, 0.67115022, 0.65392815]

# all_csm_results = [0.19499148, 0.31957274, 0.5851138,  0.76721702, 0.79237414, 0.81740726,
#  0.80528123, 0.82860888, 0.81219655, 0.76401249]

# all_ibon_results = [0.21086856, 0.41899058, 0.68185722, 0.78119856, 0.81629204, 0.86692919,
#  0.93936127, 0.9034766,  0.96718679]


# mmlu astronomy
topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
new_topns = [int((topns[k]+topns[k-1])/2) for k in range(1, len(topns))]

topns = [x // 1000 for x in topns]  # Convert to 'k'
new_topns = [x // 1000 for x in new_topns]  # Convert to 'k'

all_best_results = [1.01682651,  2.82079677,  4.58878748,  6.69836583, 11.13956404,  9.665679,
  6.50370506,  3.7653633,   3.16693368,  2.40857066,  1.9934751]

all_vanilla_results = [0.23872884,  0.53000563, -0.28462441,  0.47384529,  1.15997932, -0.15705453,
 -0.18655786,  0.68009087,  0.81831825,  0.73129763,  0.82857357]

all_avg_results = [0.20773392, -0.0210924,   0.03242596,  0.99741291,  1.13988436, -0.70852817,
  0.01110059,  0.94145783,  0.93695023,  0.82248549,  0.85377977]

all_csm_results = [0.74212576, 1.3683079,  1.4199433,  2.9180208,  4.77446137, 2.02501853,
 2.1360746,  1.36582881, 1.30443543, 1.10664966, 1.0244755]

all_ibon_results = [0.33087796, 1.48993281, 1.54749843, 2.06501447, 3.21564679, 2.09287475,
 1.50331692, 2.04600384, 1.72384632, 1.50699377]

print('all_best_results: ', all_best_results)
print('all_vanilla_results: ', all_vanilla_results)
print('all_avg_results: ', all_avg_results)
print('all_csm_results: ', all_csm_results)
print('all_ibon_results: ', all_ibon_results)
plt.plot(topns, all_vanilla_results, label='Single Query', marker='o')
plt.plot(topns, all_avg_results, label='Averaging', marker='o')
plt.plot(topns, all_best_results, label='BoN', marker='o')
plt.plot(topns, all_csm_results, label='BoN-CSM', marker='o')
plt.plot(new_topns, all_ibon_results, label='iBoN', marker='o')
plt.ylim(-2.1, 2.1)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=1, linestyle='--', color='gray')
plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
# plt.savefig(f'figures/ioi_nfs.pdf', bbox_inches='tight')
plt.savefig(f'figures/mmlu_astronomy_nfs.pdf', bbox_inches='tight')