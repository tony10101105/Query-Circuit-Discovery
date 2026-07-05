import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import matplotlib.pyplot as plt


topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
models = "GPT-2 Small (80M)"

### capability circuit
capability_eapig_5 = [-0.06, 0.19, 0.38, 0.57, 0.64, 0.68, 0.68, 0.65, 0.68, 0.69]
plt.plot(topns, capability_eapig_5, label='Capability Circuit by EAP-IG (step=5)', color='blue', linestyle='-', marker='o')

### query circuit
query_eapig_5 = [0.07, 0.08, 0.22, 0.43, 0.46, 0.50, 0.60, 0.64, 0.65, 0.68]
plt.plot(topns, query_eapig_5, label='Query Circuit by EAP-IG (step=5)', color='orange', linestyle='-', marker='o')

plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize=15)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/query_circuit_vs_task_circuit_performance.pdf', bbox_inches='tight')
