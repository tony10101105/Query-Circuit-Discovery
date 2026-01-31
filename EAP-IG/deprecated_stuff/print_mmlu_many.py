import os, json
import numpy as np
import matplotlib.pyplot as plt


errorbar = False


# with open(f'preprocessed_data/mmlu_marketing_20steps_random_pick.json', 'r') as f:
#     random_pick = json.load(f)

# random_pick1 = np.array([d['circuit_faithfulness'] for d in random_pick])
# random_pick1_mean = random_pick1.mean(axis=0)
# random_pick1_std = random_pick1.std(axis=0)
# ran = random_pick[0]['topns'][-1] - random_pick[0]['topns'][0]
# random_pick1_auc = [np.trapz(y, random_pick[0]['topns'])/ran for y in random_pick1]
# random_pick1_auc = sum(random_pick1_auc) / len(random_pick1_auc)
# random_pick1 = np.mean(random_pick1, axis=0)
# random_pick1 = random_pick1.tolist()
# print('random_pick1: ', random_pick1[:10])


with open(f'preprocessed_data/mmlu_marketing_eap-ig-inputs_metric8_20steps.json', 'r') as f:
    one_sample_data1 = json.load(f)

faithfulness1 = np.array([d['circuit_faithfulness'] for d in one_sample_data1])
faithfulness1_mean = faithfulness1.mean(axis=0)
faithfulness1_std = faithfulness1.std(axis=0)
ran = one_sample_data1[0]['topns'][-1] - one_sample_data1[0]['topns'][0]
faithfulness1_auc = [np.trapz(y, one_sample_data1[0]['topns'])/ran for y in faithfulness1]
faithfulness1_auc = sum(faithfulness1_auc) / len(faithfulness1_auc)
faithfulness1 = np.mean(faithfulness1, axis=0)
faithfulness1 = faithfulness1.tolist()

with open(f'preprocessed_data/mmlu_marketing_eap-ig-inputs_metric8_5steps.json', 'r') as f:
    one_sample_data2 = json.load(f)

faithfulness2 = np.array([d['circuit_faithfulness'] for d in one_sample_data2])
faithfulness2_mean = faithfulness2.mean(axis=0)
faithfulness2_std = faithfulness2.std(axis=0)
faithfulness2_auc = [np.trapz(y, one_sample_data2[0]['topns'])/ran for y in faithfulness2]
faithfulness2_auc = sum(faithfulness2_auc) / len(faithfulness2_auc)
faithfulness2 = np.mean(faithfulness2, axis=0)
faithfulness2 = faithfulness2.tolist()

with open(f'preprocessed_data/mmlu_marketing_eap_metric8.json', 'r') as f:
    one_sample_data3 = json.load(f)

faithfulness3 = np.array([d['circuit_faithfulness'] for d in one_sample_data3])
faithfulness3_mean = faithfulness3.mean(axis=0)
faithfulness3_std = faithfulness3.std(axis=0)
faithfulness3_auc = [np.trapz(y, one_sample_data3[0]['topns'])/ran for y in faithfulness3]
faithfulness3_auc = sum(faithfulness3_auc) / len(faithfulness3_auc)
faithfulness3 = np.mean(faithfulness3, axis=0)
faithfulness3 = faithfulness3.tolist()

# with open(f'preprocessed_data/mmlu_marketing_eap-ig-inputs_metric8_20steps_gpt4o-mini_paraphrase_all_query.json', 'r') as f:
#     one_sample_data4 = json.load(f)

# faithfulness4 = np.array([d['circuit_faithfulness'] for d in one_sample_data4])
# faithfulness4_mean = faithfulness4.mean(axis=0)
# faithfulness4_std = faithfulness4.std(axis=0)
# faithfulness4_auc = [np.trapz(y, one_sample_data4[0]['topns'])/ran for y in faithfulness4]
# faithfulness4_auc = sum(faithfulness4_auc) / len(faithfulness4_auc)
# faithfulness4 = np.mean(faithfulness4, axis=0)
# faithfulness4 = faithfulness4.tolist()
# print('faithfulness4: ', faithfulness4[:10])

# with open(f'preprocessed_data/mmlu_marketing_eap-ig-inputs_metric8_20steps_gpt4o-mini_paraphrase_only_stem.json', 'r') as f:
#     one_sample_data5 = json.load(f)

# faithfulness5 = np.array([d['circuit_faithfulness'] for d in one_sample_data5])
# faithfulness5_mean = faithfulness5.mean(axis=0)
# faithfulness5_std = faithfulness5.std(axis=0)
# faithfulness5_auc = [np.trapz(y, one_sample_data1[0]['topns'])/ran for y in faithfulness5]
# faithfulness5_auc = sum(faithfulness5_auc) / len(faithfulness5_auc)
# faithfulness5 = np.mean(faithfulness5, axis=0)
# faithfulness5 = faithfulness5.tolist()
# print('faithfulness5: ', faithfulness5[:10])

for data, mean, std, auc, label in [
    # (random_pick, random_pick1_mean, random_pick1_std, random_pick1_auc, 'Random Pick'),
    (one_sample_data1, faithfulness1_mean, faithfulness1_std, faithfulness1_auc, 'EAP-IG (step=20)'),
    (one_sample_data2, faithfulness2_mean, faithfulness2_std, faithfulness2_auc, 'EAP-IG (step=5)'),
    (one_sample_data3, faithfulness3_mean, faithfulness3_std, faithfulness3_auc, 'EAP'),
    # (one_sample_data4, faithfulness4_mean, faithfulness4_std, faithfulness4_auc, 'GPT4o-mini Paraphrase - All Query'),
    # (one_sample_data5, faithfulness5_mean, faithfulness5_std, faithfulness5_auc, 'GPT4o-mini Paraphrase - Only Stem'),
]:
    if errorbar:
        plt.errorbar(
            data[0]['topns'], mean, yerr=std,
            label=f'{label}. {auc:.2f}', fmt='-o', capsize=3
        )
    else:
        plt.plot(
            data[0]['topns'], mean,
            marker='o', linestyle='-',
            label=f'{label}. {auc:.2f}'
        )

plt.ylim(-0.1, 1.1)
# plt.xscale('log')
plt.xlabel('Edges of Edges')
plt.ylabel('Normalized Deviation Faithfulness (NDF)')
plt.legend()
plt.savefig(f'test.pdf')