import os as _os, sys as _sys
_base = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_base)
_sys.path.insert(0, _base)

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


topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
topns = [x // 1000 for x in topns]  # Convert to 'k'

# # :60
# all_best_results = [0.21350914 0.49084    0.55322371 0.59069876 0.53698564 0.66593995                          
#  0.70508488 0.83708062 0.87610718 0.89529743 0.94944221]

# all_vanilla_results = [0.12797445 0.14002619 0.17844371 0.14115528 0.11072107 0.15332686
#  0.24403005 0.50665059 0.62711627 0.66727866 0.78573656]

# all_best_random = [0.13037241 0.1303967  0.13063969 0.13137851 0.13612282 0.14895849
#  0.30022644 0.52596371 0.59892628 0.58672304 0.55679378]

# all_best_perturb_all_var1 (0.01) = [0.20964556 0.40238472 0.48617727 0.53897451 0.60098757 0.59583361
#  0.58724919 0.73374922 0.70681441 0.76528166 0.83385637]

# all_best_perturb_all_var2 (0.001) = [0.16049739 0.23513704 0.42008903 0.37408744 0.41970889 0.56637187
#  0.58945946 0.74175489 0.81811594 0.83564337 0.86227636]

# all_best_random_drop_all1 (0.1) = [0.2035573  0.4468215  0.53671665 0.60848872 0.56462263 0.58867379
#  0.57876775 0.67611723 0.72529986 0.76558483 0.8053938 ]

# all_best_random_drop_all2 (0.3) = [0.1881585  0.43128863 0.58034108 0.52283936 0.59669229 0.60341395
#  0.58847856 0.72811134 0.73656674 0.74362883 0.82049736]

# # 60:110
# all_best_results = [0.26712756 0.55079344 0.5807745  0.60506464 0.60702656 0.59005653                          
#  0.69424687 0.77699913 0.81840883 0.89250319 0.92361223]

# all_vanilla_results = [0.12901165 0.19934554 0.22101185 0.17270701 0.11860194 0.1903936
#  0.23724727 0.32527975 0.51706722 0.63513866 0.74610944]

# all_best_random = [0.1426785  0.14273138 0.142941   0.14438741 0.15308615 0.18958443
#  0.36709969 0.55163447 0.52818725 0.64012124 0.5165025 ]

# all_best_perturb_all_var1 = [0.25640052 0.49627902 0.55573763 0.59084757 0.64188262 0.65606114
#  0.66006026 0.62638118 0.68316644 0.74330032 0.78953009]

# all_best_perturb_all_var2 = [0.169723   0.3240388  0.40679651 0.50604609 0.52433929 0.50581653
#  0.58794401 0.5998207  0.74246922 0.82505089 0.87358633]

# all_best_random_drop_all1 = [0.26761619 0.51150203 0.62327805 0.58542386 0.50601962 0.48049689
#  0.60185992 0.58454168 0.70054938 0.70236066 0.79925775]

# all_best_random_drop_all2 = [0.21817487 0.51892419 0.56214927 0.60279611 0.52089791 0.50324599
#  0.59675155 0.58618874 0.67277988 0.73699304 0.81332814]

# # 110:
# all_best_results = [0.37109542 0.52269871 0.54330834 0.54885232 0.56548356 0.58942824                          
#  0.66847209 0.82935088 0.88777716 0.90652783 0.92699975]

# all_vanilla_results = [0.26141663 0.14741814 0.18618865 0.15455671 0.14746599 0.13223566
#  0.16799846 0.34877841 0.55358778 0.65487555 0.72164776]

# all_best_random = [0.22291183 0.22297178 0.22339848 0.22450975 0.23499866 0.26607635
#  0.44152178 0.61905149 0.52859935 0.61482597 0.60374525]

# all_best_perturb_all_var1 = [0.35828115 0.46931129 0.54367991 0.58041133 0.62761008 0.61889149
#  0.64491089 0.66613999 0.68662669 0.75659239 0.80112085]

# all_best_perturb_all_var2 = [0.28143779 0.31336371 0.40973527 0.48756637 0.56368446 0.54671833
#  0.59447138 0.70235683 0.76473603 0.7946602  0.84538138]

# all_best_random_drop_all1 = [0.37222862 0.52730866 0.52434375 0.56669445 0.47969633 0.55129953
#  0.53911871 0.60207218 0.7016956  0.74707147 0.74643169]

# all_best_random_drop_all2 = [0.32716662 0.57157259 0.53844746 0.60731948 0.59918878 0.60009842
#  0.5844191  0.6056428  0.69019027 0.76395164 0.75396238]

# averaged
all_best = [0.2747, 0.5194, 0.5595, 0.5839, 0.5679, 0.6198, 0.6914, 0.8152, 0.8604, 0.8975, 0.9347]

all_vanilla = [0.1652, 0.1616, 0.1946, 0.1552, 0.1235, 0.1597, 0.2208, 0.4034, 0.5706, 0.6533, 0.7550]

all_best_random = [0.1600, 0.1600, 0.1603, 0.1614, 0.1690, 0.1947, 0.3613, 0.5601, 0.5562, 0.6121, 0.5565]

all_best_perturb_all_var1 = [0.2661, 0.4518, 0.5249, 0.5675, 0.6218, 0.6220, 0.6271, 0.6797, 0.6935, 0.7556, 0.8102]

all_best_perturb_all_var2 = [0.1969, 0.2860, 0.4129, 0.4489, 0.4939, 0.5410, 0.5903, 0.6842, 0.7785, 0.8208, 0.8613]

all_best_random_drop_all1 = [0.2712, 0.4903, 0.5618, 0.5894, 0.5219, 0.5428, 0.5754, 0.6255, 0.7106, 0.7397, 0.7871]

all_best_random_drop_all2 = [0.2364, 0.4989, 0.5628, 0.5725, 0.5724, 0.5695, 0.5901, 0.6476, 0.7028, 0.7471, 0.7998]

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
plt.savefig(f'figures/mmlu_astronomy_different_bon.pdf', bbox_inches='tight')