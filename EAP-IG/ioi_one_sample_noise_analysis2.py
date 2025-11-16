from functools import partial

import os
import sys
import ast
import json
import numpy as np
import random
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute
from src.eap.utils import topn_indices, set_seed

os.environ["TRANSFORMERS_CACHE"] = "/data/huggingface"

set_seed(2025)

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, data_num):
        self.df = pd.read_csv(filepath)
        self.df = self.df[:data_num]

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], [row['correct_idx'], row['incorrect_idx']]
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


data_num = 1
topns = [50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small' # meta-llama/Llama-3.2-1B-Instruct, gpt2-small
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/ioi_gpt2.csv', data_num=data_num)
dataloader = ds.to_dataloader(batch_size=1)

from scipy.stats import kendalltau
var = 0.005
purturb_times = 50
indices = np.linspace(0, var, purturb_times)
# model_perform = [-2.3826589584350586, -2.3834495544433594, -2.386913299560547, -2.390085220336914, -2.3771371841430664, -2.3745193481445312, -2.380459785461426, -2.3628711700439453, -2.391965866088867, -2.349747657775879, -2.388948440551758, -2.402646064758301, -2.368950843811035, -2.359498977661133, -2.4048757553100586, -2.3971996307373047, -2.3760461807250977, -2.371354103088379, -2.35054874420166, -2.3543567657470703, -2.3445863723754883, -2.350172996520996, -2.3786354064941406, -2.3542442321777344, -2.3662986755371094, -2.3941993713378906, -2.350632667541504, -2.457958221435547, -2.3849058151245117, -2.341153144836426, -2.482034683227539, -2.3282251358032227, -2.36013126373291, -2.4258861541748047, -2.3463621139526367, -2.4038591384887695, -2.447481155395508, -2.367403984069824, -2.48244571685791, -2.3417835235595703, -2.4377546310424805, -2.290196418762207, -2.254842758178711, -2.2899694442749023, -2.223154067993164, -2.3897523880004883, -2.274754524230957, -2.2679309844970703, -2.4077224731445312, -2.210982322692871]
model_perform = [-2.3826589584350586, -2.383063316345215, -2.38478946685791, -2.386404037475586, -2.3798999786376953, -2.3785934448242188, -2.381580352783203, -2.372799873352051, -2.3873443603515625, -2.3662662506103516, -2.3858470916748047, -2.392674446105957, -2.3759498596191406, -2.3711891174316406, -2.3939342498779297, -2.3899402618408203, -2.3795690536499023, -2.3771419525146484, -2.366586685180664, -2.368814468383789, -2.3636417388916016, -2.366419792175293, -2.380849838256836, -2.3684263229370117, -2.3748302459716797, -2.388570785522461, -2.3670520782470703, -2.420469284057617, -2.3834896087646484, -2.3624162673950195, -2.4327688217163086, -2.3559303283691406, -2.371990203857422, -2.4040937423706055, -2.365666389465332, -2.3936586380004883, -2.4153757095336914, -2.3752899169921875, -2.433291435241699, -2.3619766235351562, -2.410811424255371, -2.337765693664551, -2.31917667388916, -2.335927963256836, -2.3027305603027344, -2.3858394622802734, -2.328244209289551, -2.326540946960449, -2.395085334777832, -2.29705810546875]
model_perform = [-1*i for i in model_perform]
taus = []
para_data = []
initial_scores = np.load(f"score_data/ioi_1st_sample_noise/var_0.005_0.0.npy")
for v in indices[1:]:
    scores = np.load(f"score_data/ioi_1st_sample_noise/var_0.005_{v}.npy")
    para_data.append(scores)

    tau, p_value = kendalltau(scores[np.isfinite(scores)], initial_scores[np.isfinite(initial_scores)])
    taus.append(tau)
    
fig, ax1 = plt.subplots()

# First axis: model performance
ax1.plot(indices[1:], model_perform[1:], label='Performance', marker='o', color='tab:blue')
ax1.set_xlabel('var')
ax1.set_ylabel('Performance', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Second axis: tau
ax2 = ax1.twinx()
ax2.plot(indices[1:], taus, label='Kendall Tau', marker='s', color='tab:orange')
ax2.set_ylabel('Tau', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Optional: Title and layout
fig.suptitle('Performance vs. Kendall Tau')
fig.tight_layout()
plt.legend()
plt.savefig(f'test.pdf', bbox_inches='tight')
exit(0)

para_data = np.stack(para_data, axis=0)   # shape: (len(arrays), rows, cols)

all_best_results = []
all_vanilla_results = []
all_avg_results = []
all_csm_results = []
all_ibon_results = []


for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    single_data = [(clean, corrupted, label)]

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, manual_pad=True).mean().item()
    # print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')
    
    best_results = [-1]*len(topns)
    best_para_indices = [0]*len(topns) # keep track of the best paraphrase index for each topn

    all_indices = list(range(para_data.shape[0]))
    available_idxs = [para_idx for para_idx in all_indices if para_idx != i]
    sampled = random.sample(available_idxs, 9)
    sampled = [i] + sampled
    
    for j in tqdm(sampled, total=len(sampled), desc="Processing paraphrases"):
        model.reset_hooks()
        
        g = Graph.from_model(model)

        g.scores = torch.from_numpy(para_data[j])

        circuit_faithfulness = []
        for topn in topns:
            g.apply_topn(topn, True)

            print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

            results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
            results = results.mean().item()
            
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

            print(f"Model performance: {baseline:.2f}; corrupted baseline: {corrupted_baseline:.2f}'; circuit performance: {results:.2f}; faithfulness: {faithfulness:.2f}")

        if j == i:
            all_vanilla_results.append(circuit_faithfulness)

        for idx in range(len(best_results)):
            if circuit_faithfulness[idx] > best_results[idx]:
                best_results[idx] = circuit_faithfulness[idx]
                best_para_indices[idx] = j

    all_best_results.append(best_results)

    # averaging
    model.reset_hooks()
    g = Graph.from_model(model)

    g.scores = torch.from_numpy(para_data[sampled].mean(0))

    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)

    all_avg_results.append(circuit_faithfulness)
    
    # BoN with Constraint-adaptive Score Matrix (BoN-CSM)
    score_mat = np.full_like(para_data[j], -np.inf, dtype=float)
    tier_mat  = np.full(para_data[j].shape, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
    filled    = np.zeros_like(score_mat, dtype=bool)

    for l, topn in enumerate(topns):           # l = 0 (highest priority), 1, ...
        best_para_idx = best_para_indices[l]
        M = np.abs(para_data[best_para_idx])
        M = np.where(np.isfinite(M), M, -np.inf)
        for (a, b) in topn_indices(M, topn):
            if not filled[a, b]:
                score_mat[a, b] = M[a, b]
                tier_mat[a, b]  = l
                filled[a, b]    = True
    
    model.reset_hooks()
    g = Graph.from_model(model)
    g.scores = torch.from_numpy(score_mat)

    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn_by_tier(topn, tier_mat)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)
    all_csm_results.append(circuit_faithfulness)
    
    # interpolated BoN (iBoN)
    new_topns = [int((topns[k]+topns[k-1])/2) for k in range(1, len(topns))]
    
    circuit_faithfulness = []
    for k, topn in enumerate(new_topns):
        model.reset_hooks()
        g = Graph.from_model(model)
    
        previous_topn, next_topn = topns[k], topns[k+1]

        score_mat = np.full_like(para_data[j], -np.inf, dtype=float)
        tier_mat  = np.full(para_data[j].shape, fill_value=np.iinfo(np.int32).max, dtype=np.int32)
        filled    = np.zeros_like(score_mat, dtype=bool)

        for l, top in zip([k, k+1], [previous_topn, next_topn]):
            best_para_idx = best_para_indices[l]
            M = np.abs(para_data[best_para_idx])
            M = np.where(np.isfinite(M), M, -np.inf)
            for (a, b) in topn_indices(M, top):
                if not filled[a, b]:
                    score_mat[a, b] = M[a, b]
                    tier_mat[a, b]  = l
                    filled[a, b]    = True
        
        g.scores = torch.from_numpy(score_mat)
        g.apply_topn_by_tier(topn, tier_mat)
        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention, quiet=True)
        results = results.mean().item()
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)
    all_ibon_results.append(circuit_faithfulness)
    
print('topns: ', topns)

all_best_results = np.array(all_best_results).mean(0)
all_vanilla_results = np.array(all_vanilla_results).mean(0)
all_avg_results = np.array(all_avg_results).mean(0)
all_csm_results = np.array(all_csm_results).mean(0)
all_ibon_results = np.array(all_ibon_results).mean(0)
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
plt.xlabel('Number of Edges')
plt.ylabel('Normalized Deviation Faithfulness (NDF)')
plt.legend()
plt.savefig(f'ioi_20steps_noise.pdf', bbox_inches='tight')
plt.close()