import matplotlib.pyplot as plt
import numpy as np


topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
new_topns = [int((topns[k]+topns[k-1])/2) for k in range(1, len(topns))]
topns = [x // 1000 for x in topns]  # Convert to 'k'
new_topns = [x // 1000 for x in new_topns]  # Convert to 'k'

# :400
all_best_results1 = [0.2404873, 0.47163847, 0.57003145, 0.56067954, 0.54942843, 0.56430236,
 0.66588209, 0.77642331, 0.8365758,  0.89153645, 0.91787559]
all_vanilla_results1 = [0.13680736, 0.18101937, 0.16460078, 0.15939428, 0.13687792, 0.14518492,
 0.23341049, 0.40911673, 0.56088467, 0.65104955, 0.75131458]
all_avg_results1 = [0.13741628, 0.15449506, 0.16913071, 0.15318504, 0.14559585, 0.14488569,
 0.22029281, 0.37798545, 0.53818049, 0.62026811, 0.71485685]
all_csm_results1 = [0.2404873, 0.28754142, 0.27894018, 0.22114171, 0.16284102, 0.16500438,
 0.3612304, 0.57374699, 0.69357602, 0.79093265, 0.88480441]
all_ibon_results1 = [0.17486188, 0.20570854, 0.22256401, 0.18011836, 0.17140502, 0.25669801,
 0.48438541, 0.60847644, 0.73358523, 0.82436216]

# 400:800
all_best_results2 = [0.26371098, 0.49114654, 0.56490254, 0.58030875, 0.57657873, 0.60380213,
 0.70430514, 0.81704271, 0.87470178, 0.91724729, 0.94091221]
all_vanilla_results2 = [0.15348534, 0.18950846, 0.18400468, 0.16352584, 0.15082209, 0.15419245,
 0.28499534, 0.43468689, 0.56889176, 0.67931521, 0.7745076]
all_avg_results2 = [0.14976477, 0.17719296, 0.18267247, 0.15464811, 0.14998628, 0.16627411,
 0.28058865, 0.41537921, 0.54506002, 0.66572796, 0.76241143]
all_csm_results2 = [0.26371098, 0.29615993, 0.24560149, 0.23437938, 0.20092009, 0.20979964,
 0.39072141, 0.59349636, 0.73214276, 0.812988, 0.90922207]
all_ibon_results2 = [0.18454649, 0.20682642, 0.21697341, 0.16354059, 0.17997948, 0.27386404,
 0.50200602, 0.67644796, 0.75805737, 0.8472288]

# 800:
all_best_results3 = [0.25766004, 0.49825382, 0.5746447, 0.59156571, 0.59106104, 0.58270435,
 0.68755629, 0.77870997, 0.85261959, 0.89938768, 0.93780313]
all_vanilla_results3 = [0.15173582, 0.16208473, 0.15175943, 0.16285353, 0.151736, 0.13839633,
 0.22792712, 0.39999931, 0.54162931, 0.64596622, 0.75034822]
all_avg_results3 = [0.13507323, 0.16281783, 0.17163802, 0.16044457, 0.14604339, 0.15980867,
 0.22989998, 0.38396887, 0.52682373, 0.61889816, 0.71319398]
all_csm_results3 = [0.25766004, 0.31789989, 0.27957154, 0.21449061, 0.21791494, 0.18865435,
 0.33854276, 0.55745464, 0.69497374, 0.81176363, 0.90366011]
all_ibon_results3 = [0.19334784, 0.21054102, 0.23624781, 0.17519248, 0.19267857, 0.23216872,
 0.46218296, 0.62962219, 0.73604941, 0.85044011]

all_best_results = [(400*x+400*y+372*z)/1172 for x,y,z in zip(all_best_results1, all_best_results2, all_best_results3)]
all_vanilla_results = [(400*x+400*y+372*z)/1172 for x,y,z in zip(all_vanilla_results1, all_vanilla_results2, all_vanilla_results3)]
all_avg_results = [(400*x+400*y+372*z)/1172 for x,y,z in zip(all_avg_results1, all_avg_results2, all_avg_results3)]
all_csm_results = [(400*x+400*y+372*z)/1172 for x,y,z in zip(all_csm_results1, all_csm_results2, all_csm_results3)]
all_ibon_results = [(400*x+400*y+372*z)/1172 for x,y,z in zip(all_ibon_results1, all_ibon_results2, all_ibon_results3)]
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
plt.savefig(f'figures/arc_challenge.pdf')