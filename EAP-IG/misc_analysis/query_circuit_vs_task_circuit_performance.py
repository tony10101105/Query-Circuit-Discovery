import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import numpy as np


topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
models = "GPT-2 Small (80M)"

### capability circuit
# capability_eap = [0, 0, 0, 0.52, 0.52, 0.63, 0.61, 0.63, 0.67, 0.64]
capability_eapig_5 = [-0.06, 0.19, 0.38, 0.57, 0.64, 0.68, 0.68, 0.65, 0.68, 0.69]
# capability_eapig_20 = [-0.06, 0.20, 0.44, 0.48, 0.68, 0.67, 0.63, 0.62, 0.74, 0.65]

# plt.plot(topns, capability_eap, label='EAP', color='blue', linestyle='--', marker='o')
plt.plot(topns, capability_eapig_5, label='Capability Circuit by EAP-IG (step=5)', color='blue', linestyle='-', marker='o')
# plt.plot(topns, capability_eapig_20, label='EAP-IG (step=20)', color='blue', linestyle='--', marker='o')

### query circuit
# query_eap = [0.03, 0.08, 0.22, 0.43, 0.53, 0.60, 0.62, 0.65, 0.66, 0.71]
query_eapig_5 = [0.07, 0.08, 0.22, 0.43, 0.46, 0.50, 0.60, 0.64, 0.65, 0.68]
# query_eapig_20 = [0.08, 0.10, 0.26, 0.45, 0.44, 0.54, 0.58, 0.64, 0.64, 0.65]

# plt.plot(topns, query_eap, label='EAP', color='orange', linestyle='--', marker='o')
plt.plot(topns, query_eapig_5, label='Query Circuit by EAP-IG (step=5)', color='orange', linestyle='-', marker='o')
# plt.plot(topns, query_eapig_20, label='EAP-IG (step=20)', color='orange', linestyle='--', marker='o')

plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize=15)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/query_circuit_vs_task_circuit_performance.pdf', bbox_inches='tight')
