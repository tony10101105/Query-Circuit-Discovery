import os, json
import numpy as np
import matplotlib.pyplot as plt


with open(f'preprocessed_data/mmlu_marketing_eap-ig-inputs_metric3_one_sample_per_circuit.json', 'r') as f:
    one_sample_data3 = json.load(f)

one_sample_faithfulness3 = [d['circuit_faithfulness'] for d in one_sample_data3]
one_sample_faithfulness3 = np.mean(one_sample_faithfulness3, axis=0)
one_sample_faithfulness3 = one_sample_faithfulness3.tolist()

bs = [d['baseline'] for d in one_sample_data3]
bs = np.mean(bs, axis=0).tolist()
print('baseline: ', bs)

cbs = [d['corrupted_baseline'] for d in one_sample_data3]
cbs = np.mean(cbs, axis=0).tolist()
print('corrupted baseline: ', cbs)

cr = [d['circuit_faithfulness'] for d in one_sample_data3]
cr = np.mean(cr, axis=0).tolist()
print('circuit_results: ', cr)

plt.plot(one_sample_data3[0]['topns'], one_sample_faithfulness3, label='One Sample Metric 3', marker='o')
plt.ylim(-2.5, 2.1)
plt.xscale('log')
plt.xlabel('Top-K Edges')
plt.ylabel('Circuit Faithfulness')
plt.title(f'MMLU Circuit Faithfulness vs Top-K Edges')
plt.legend()

plt.savefig(f'test.png', dpi=500)