from functools import partial

import ast
import json
import matplotlib.pyplot as plt
import pygraphviz
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute 
from utils import get_logit_positions, logit_diff, EAPDataset
def prob_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    incorrect_label = [j for i, j in labels]
    # label to tensor
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)
    # print('before logits: ', logits.shape) # torch.Size([bs, n_pos, 50257])
    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample
    probs = torch.softmax(logits, dim=-1)
    
    # print(correct_label.shape) # torch.Size([bs])
    good = torch.gather(probs, -1, correct_label).squeeze(1)
    
    incorrect_label = torch.tensor(incorrect_label, device=logits.device)

    bad = torch.gather(probs, -1, incorrect_label).mean(dim=-1)

    results = good - bad

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


topns = [200, 750, 1500, 3000, 5000, 10000, 20000, 30000] # 32491
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda', dtype=torch.float16)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_attention = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/mmlu_Llama-32-3B.csv', category='marketing', num_samples=500, mc=True)
dataloader = ds.to_dataloader(batch_size=1)

g = Graph.from_model(model)

print('evaluating baseline...')
baseline = evaluate_baseline(model, dataloader, partial(logit_diff, mc=True, loss=False, mean=False)).mean().item()
corrupted_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, mc=True, loss=False, mean=False), run_corrupted=True).mean().item()
print(f"Original performance was {baseline}; the corrupted_baseline performance is {corrupted_baseline}")

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
print('attributing...')
attribute(model, g, dataloader, partial(logit_diff, mc=True, loss=True, mean=True), method=method, ig_steps=5, intervention=intervention)

print('evaluating circuit...')
circuit_results = []
circuit_faithfulness = []
for topn in topns:
    g.apply_topn(topn, True)
    g.prune()
    print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

    # g.to_json(f'ioi_{model_name}_{topn}_circuit.json')
    # gz = g.to_image(f'ioi_{model_name}_{topn}_circuit.png')

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, mc=True, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)
    results = results.mean().item()
    circuit_results.append(results)
    
    faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
    circuit_faithfulness.append(faithfulness)
    
    print(f"Original performance was {baseline}; the circuit's performance is {results}; faithfulness is {faithfulness}")

all_results = {
    'baseline': baseline,
    'corrupted_baseline': corrupted_baseline,
    'topns': topns,
    'circuit_results': circuit_results,
    'circuit_faithfulness': circuit_faithfulness
}
# with open(f'mmlu_{method.lower()}_all_sample_data.json', 'w') as f:
#     json.dump(all_results, f, indent=2)

"""
# plt.plot(topns, circuit_results, label=method, marker='o')  # marker adds dots on points
plt.plot(topns, circuit_faithfulness, label=method, marker='o')  # marker adds dots on points

plt.ylim(-0.1, 1.1)
plt.xlim(0, max(topns)+200)
plt.xlabel('Top-K Edges')
plt.ylabel('Circuit Faithfulness')
plt.title('IOI Circuit Faithfulness vs Top-K Edges')
plt.legend()

plt.savefig('oseap_figures/ioi_circuit_faithfulness.png')
"""