import matplotlib.pyplot as plt
import numpy as np


topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
topns = [x // 1000 for x in topns]  # Convert to 'k'

# BoN: 20056.72s

# Total time: 447.79 seconds
random = [0.15997711, 0.15998025, 0.15993742, 0.15989151, 0.15989239, 0.16299036,
 0.16722285, 0.17177814, 0.16986179, 0.14237804, 0.13655816]

# eap-ig 5 steps:
eapig5 = [0.16518782, 0.16158165, 0.19458644, 0.15523717, 0.12346666, 0.15969203,
 0.22079015, 0.40336656, 0.57059884, 0.65327912, 0.75499257]

# eap-ig 20 steps:
eapig20 = [0.17991761, 0.14336368, 0.19846638, 0.1508859, 0.1417061, 0.12408019,
 0.26229217, 0.42504424, 0.53705626, 0.65992874, 0.77710112]

# # eap-ig 100 steps: 4261.50 seconds
eapig100 = [0.15721693, 0.14250772, 0.1842533, 0.19359973, 0.15944608, 0.16990736,
 0.27417702, 0.48036088, 0.57227406, 0.66510492, 0.76148926]

# eap-ig 300 steps: 11101.52
eapig300 = [0.18147012, 0.16708517, 0.16186907, 0.13981299, 0.14497318, 0.1230551,
 0.26256467, 0.39336569, 0.54141599, 0.64672503, 0.76419709]

# eap-ig 500 steps: 18269.62
eapig500 = [0.18155827, 0.15678237, 0.16907607, 0.14509803, 0.13472943, 0.12399007,
            0.25964571, 0.40208133, 0.54912595, 0.65026558, 0.75908047]

# eap-ig 1000 steps: 36096.97
eapig1000 = [0.1798534, 0.1578881, 0.17259863, 0.1353677, 0.13596156, 0.12910962,
 0.25092982, 0.38907298, 0.55398901, 0.65804941, 0.76699272]


plt.plot(topns, random, label='Random', marker='o')
plt.plot(topns, eapig5, label='EAP-IG (step=5)', marker='o')
plt.plot(topns, eapig20, label='EAP-IG (step=20)', marker='o')
plt.plot(topns, eapig100, label='EAP-IG (step=100)', marker='o')
# plt.plot(topns, eapig300, label='EAP-IG (step=300)', marker='o')
plt.plot(topns, eapig500, label='EAP-IG (step=500)', marker='o')
plt.plot(topns, eapig1000, label='EAP-IG (step=1000)', marker='o')
plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges (k)', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(f'figures/mmlu_astronomy_not_work.pdf')