
import matplotlib.pyplot as plt
import numpy as np

models = ["GPT-2 Small\n(80M)", "GPT-2 Medium\n(345M)", "GPT-2 Large\n(774M)", "GPT-2 XL\n(1.5B)"]
x = np.arange(len(models))
width = 0.2

# in seconds
eap = [0.1, 0.2, 0.3, 0.5]
eap_ig_5 = [0.4, 0.8, 1.2, 1.7]
eap_ig_20 = [1.5, 3, 4.6, 6.2]
acdc = [280, 3701, 18874, 74417]

plt.figure(figsize=(10,7.5))
plt.bar(x - 1.5*width, eap, width, label='EAP')
plt.bar(x - 0.5*width, eap_ig_5, width, label='EAP-IG (step=5)')
plt.bar(x + 0.5*width, eap_ig_20, width, label='EAP-IG (step=20)')
plt.bar(x + 1.5*width, acdc, width, label='ACDC')

plt.yscale('log')
plt.ylabel('Per-Query Runtime (Seconds)', fontsize=20)
plt.xticks(x, models, fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figures/method_runtime_time_vs_model.pdf', bbox_inches='tight')
