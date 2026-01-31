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


# mmlu astronomy
topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
new_topns = [int((topns[k]+topns[k-1])/2) for k in range(1, len(topns))]

topns = [x // 1000 for x in topns]  # Convert to 'k'
new_topns = [x // 1000 for x in new_topns]  # Convert to 'k'

all_best_results = [0.119562835, 0.15871639, 0.284334955, 0.3870329, 0.42023372,
 0.3968200, 0.3925980, 0.438858855, 0.503420705, 0.628452515, 0.717036375]

all_vanilla_results = [0.114011575, 0.11652706, 0.10453404, 0.15274115, 0.1539381,
 0.12732465, 0.127799355, 0.154512095, 0.208916655, 0.28104397, 0.344296235]

all_avg_results = [0.11618662, 0.106121465, 0.10051594, 0.12742346, 0.11709876,
 0.129995365, 0.131827805, 0.17010297, 0.202384905, 0.2719906, 0.338550545]

all_csm_results = [0.11924778, 0.13116391, 0.15353555, 0.18120357, 0.200222275,
 0.18109845, 0.193335755, 0.185622825, 0.31861645, 0.457374645, 0.56846889]

all_ibon_results = [0.115640475, 0.11773905, 0.1258856, 0.165095405, 0.147098695,
 0.176430285, 0.178023075, 0.254649395, 0.415890045, 0.500003835]

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
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/mmlu_astronomy_llama3-8b.pdf', bbox_inches='tight')