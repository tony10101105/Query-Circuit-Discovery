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


topns = [500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000] # 32491 for gpt2-small, 386713 for llama

# :100
all_best_results = [0.4035161,  0.52167704, 0.57747365, 0.61693602, 0.65134339, 0.68742944,
 0.72221025, 0.7549591,  0.74742421, 0.78039349, 0.79663469]

all_vanilla_results = [0.33824212, 0.45144525, 0.47787277, 0.51174906, 0.56015553, 0.59581666,
 0.61266267, 0.68166039, 0.67556041, 0.70836336, 0.73841475]

all_best_random = [3.16825663e-07, 1.85319357e-06, 5.19130117e-06, 4.27666802e-06,
 9.85691211e-06, 2.19400727e-04, 7.29233836e-04, 6.18984195e-03,
 6.96467089e-03, 6.43814473e-03, 1.17362165e-02]

all_best_perturb_all_var1 = [0.36682814, 0.50237301, 0.53431684, 0.58087458, 0.64617236, 0.70185596,
 0.6967691,  0.76530026, 0.76261009, 0.77605324, 0.78210905]

all_best_perturb_all_var2 = [0.35050691, 0.46458504, 0.50247694, 0.55168185, 0.59232961, 0.6322502,
 0.66823415, 0.74220229, 0.75116072, 0.77770079, 0.79476302]

all_best_random_drop_all1 = [0.35781923, 0.49688635, 0.51579741, 0.55591158, 0.6025442,  0.64990587,
 0.65908387, 0.71678858, 0.72816207, 0.75209554, 0.75844363]

all_best_random_drop_all2 = [0.35236628, 0.4634662,  0.49248721, 0.52919975, 0.58939496, 0.62217306,
 0.65785202, 0.70000897, 0.70050019, 0.73683874, 0.74609212]


# 100:300
all_best_results = [0.47635119, 0.6319552,  0.70724298, 0.73565468, 0.79030138, 0.81819434,
 0.83439997, 0.85758596, 0.87416046, 0.88517209, 0.89502234]

all_vanilla_results = [0.32899077, 0.46417949, 0.50280333, 0.53369855, 0.57647923, 0.60464961,
 0.60346347, 0.66789315, 0.70665882, 0.70736465, 0.72322338]

all_best_random = [6.81389713e-07, 1.11903346e-05, 2.86666654e-05, 5.88661074e-05,
 7.72807017e-05, 1.93869916e-04, 9.14132220e-04, 2.81610932e-03,
 6.12154699e-03, 7.83621913e-03, 1.34436773e-02]

all_best_perturb_all_var1 = [0.40549321, 0.55689269, 0.63576055, 0.67879325, 0.72155172, 0.75572047,
 0.7967509,  0.81871742, 0.83868613, 0.84026522, 0.86864329]

all_best_perturb_all_var2 = [0.35439617, 0.50227888, 0.56293067, 0.6128386,  0.66154295, 0.71805128,
 0.75708741, 0.82166906, 0.84647922, 0.87424266, 0.88198186]

all_best_random_drop_all1 = [0.36920979, 0.52619667, 0.57811826, 0.63184831, 0.68232252, 0.76334538,
 0.76234109, 0.80918459, 0.81088198, 0.82783185, 0.83847894]

all_best_random_drop_all2 = [0.35189079, 0.49982496, 0.55031593, 0.60136208, 0.64239802, 0.69365519,
 0.69001831, 0.7295089,  0.76666807, 0.77595764, 0.80214411]

# 300:
all_best_results:  [0.52065882, 0.67746979, 0.74190295, 0.79330997, 0.82367581, 0.86423448,
 0.89540267, 0.90282524, 0.93217694, 0.92595461, 0.94476276]

all_vanilla_results:  [0.38894679, 0.51589294, 0.56916167, 0.58826195, 0.64064164, 0.65652719,
 0.6856638,  0.71699662, 0.74071022, 0.73166908, 0.78213864]

all_best_random:  [4.80910560e-07, 2.33370794e-06, 7.47160544e-06, 1.97714699e-05,
 6.97989850e-05, 1.97424318e-04, 9.21813056e-04, 2.67225499e-03,
 4.15985691e-03, 7.31442278e-03, 1.16825277e-02]

all_best_perturb_all_var1 = [0.4659709, 0.62089948, 0.69536819, 0.72415599, 0.78625594, 0.80454677,
 0.85357181, 0.86625672, 0.88884584, 0.88280189, 0.90375565]

all_best_perturb_all_var2 = [0.42228573, 0.56739793, 0.63685464, 0.66666022, 0.72538932, 0.78559555,
 0.84014729, 0.88418968, 0.90297149, 0.91737924, 0.92702064]

all_best_random_drop_all1 = [0.43683278, 0.56946677, 0.64958989, 0.68539072, 0.72287747, 0.80059517,
 0.82223281, 0.84339251, 0.85162233, 0.85539278, 0.87730516]

all_best_random_drop_all2 = [0.41768695, 0.53795535, 0.61606597, 0.62822059, 0.71574393, 0.70922585,
 0.76840165, 0.78901151, 0.8123789,  0.81991101, 0.83890635]

# average
all_best = [0.480, 0.628, 0.695, 0.735, 0.776, 0.810, 0.836, 0.855, 0.872, 0.881, 0.895]

all_vanilla = [0.355, 0.482, 0.524, 0.551, 0.599, 0.624, 0.638, 0.690, 0.714, 0.717, 0.750]

all_best_random = [5.3e-07, 5.8e-06, 1.5e-05, 3.2e-05, 6.1e-05, 2.0e-04, 8.8e-04, 3.4e-03, 5.5e-03, 7.3e-03, 0.0124]

all_best_perturb_all_var1 = [0.422, 0.572, 0.639, 0.677, 0.732, 0.764, 0.799, 0.827, 0.844, 0.844, 0.865]

all_best_perturb_all_var2 = [0.381, 0.521, 0.580, 0.622, 0.673, 0.728, 0.773, 0.831, 0.850, 0.872, 0.883]

all_best_random_drop_all1 = [0.394, 0.538, 0.594, 0.638, 0.683, 0.756, 0.766, 0.804, 0.811, 0.824, 0.838]

all_best_random_drop_all2 = [0.378, 0.508, 0.565, 0.598, 0.661, 0.686, 0.715, 0.747, 0.772, 0.786, 0.806]

auc_best = np.trapz(all_best, topns) / (topns[-1] - topns[0])
auc_vanilla = np.trapz(all_vanilla, topns) / (topns[-1] - topns[0])
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

plt.plot(topns, all_vanilla, label='Single Query', marker='o')
plt.plot(topns, all_best, label='BoN-Para.', marker='o')
plt.plot(topns, all_best_random, label='BoN-Random', marker='o')
plt.plot(topns, all_best_perturb_all_var1, label='BoN-GP ($\sigma$=0.01)', marker='o')
plt.plot(topns, all_best_perturb_all_var2, label='BoN-GP ($\sigma$=0.001)', marker='o')
plt.plot(topns, all_best_random_drop_all1, label='BoN-ER ($t$=0.1)', marker='o')
plt.plot(topns, all_best_random_drop_all2, label='BoN-ER ($t$=0.3)', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='lower right', fontsize=13)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/arithmeticmul_different_bon.pdf', bbox_inches='tight')