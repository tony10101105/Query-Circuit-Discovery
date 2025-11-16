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


topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491

all_best_results = [0.11563751, 0.23446632, 0.48464201, 0.6398324, 0.70688095, 0.7358685,
 0.76033412, 0.77552521, 0.78696191, 0.79829109]

all_vanilla_results = [0.0521145, 0.12157143, 0.29282145, 0.43693575, 0.50925974, 0.54703262,
 0.57694832, 0.59397097, 0.60657474, 0.62499063]

all_best_random = [4.12322147e-07, 7.99273290e-07, 1.90168194e-05, 1.57246354e-04,
 5.96258898e-04, 1.22007694e-03, 1.82803319e-03, 2.80330624e-03,
 4.08385938e-03, 4.69105398e-03]

all_best_perturb_all_var1 = [0.06922366, 0.1651486,  0.38840359, 0.56596025, 0.63758983, 0.6829509,
 0.70891607, 0.72133754, 0.73617505, 0.74885158]

all_best_perturb_all_var2 = [0.05435138, 0.13021225, 0.32039806, 0.48440413, 0.56466538, 0.61302542,
 0.65050596, 0.6736293,  0.69223447, 0.7131768]

all_best_random_drop_all1 = [0.08083982, 0.18881549, 0.40515769, 0.58674474, 0.65261776, 0.69602565,
 0.72559, 0.73748208, 0.7515141,  0.76076399]

all_best_random_drop_all2 = [0.07459042, 0.16573035, 0.34155434, 0.48585965, 0.55209067, 0.59279448,
 0.6187435,  0.63832053, 0.64539848, 0.66351276]

auc_best = np.trapz(all_best_results, topns) / (topns[-1] - topns[0])
auc_vanilla = np.trapz(all_vanilla_results, topns) / (topns[-1] - topns[0])
auc_random = np.trapz(all_best_random, topns) / (topns[-1] - topns[0])
auc_perturb_all_var1 = np.trapz(all_best_perturb_all_var1, topns) / (topns[-1] - topns[0])
auc_perturb_all_var2 = np.trapz(all_best_perturb_all_var2, topns) / (topns[-1] - topns[0])
auc_random_drop_all1 = np.trapz(all_best_random_drop_all1, topns) / (topns[-1] - topns[0])
auc_random_drop_all2 = np.trapz(all_best_random_drop_all2, topns) / (topns[-1] - topns[0])
print(f'BoN-Para. AUC: {auc_best}')
print(f'Single Query AUC: {auc_vanilla}')
print(f'BoN-Random AUC: {auc_random}')
print(f'BoN-SP (0.01) AUC: {auc_perturb_all_var1}')
print(f'BoN-SP (0.001) AUC: {auc_perturb_all_var2}')
print(f'BoN-ER (0.1) AUC: {auc_random_drop_all1}')
print(f'BoN-ER (0.3) AUC: {auc_random_drop_all2}')

plt.plot(topns, all_vanilla_results, label='Single Query', marker='o')
plt.plot(topns, all_best_results, label='BoN-Para.', marker='o')
plt.plot(topns, all_best_random, label='BoN-Random', marker='o')
plt.plot(topns, all_best_perturb_all_var1, label='BoN-GP ($\sigma$=0.01)', marker='o')
plt.plot(topns, all_best_perturb_all_var2, label='BoN-GP ($\sigma$=0.001)', marker='o')
plt.plot(topns, all_best_random_drop_all1, label='BoN-ER ($t$=0.1)', marker='o')
plt.plot(topns, all_best_random_drop_all2, label='BoN-ER ($t$=0.3)', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=13)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/ioi_different_bon.pdf', bbox_inches='tight')