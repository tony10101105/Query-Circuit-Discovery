from functools import partial

import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # For TransformerLens:
    # Optional: sets the default generator for dropout, noise etc.
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
set_seed(2025) 

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
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
        return row['clean'], row['corrupted'], [row['clean_answer_idx'], row['corrupted_answer_idx']]
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False, a=False):
    logits = get_logit_positions(logits, input_length)
    good_bad = torch.gather(logits, -1, labels.to(logits.device))

    probs = torch.softmax(logits, dim=-1)
    good_bad = torch.gather(probs, -1, labels.to(probs.device))
    if a:
        print('good: ', good_bad[:, 0])
        print('bad: ', good_bad[:, 1])
        print('diff: ', good_bad[:, 0] - good_bad[:, 1])

    
    results = good_bad[:, 0] - good_bad[:, 1]
    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results


data_num = 1000
# topns = [50, 100, 200, 300, 400, 500, 750, 1000] # 32491
topns = [100]
assert len(topns) == 1
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 5
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset('probing_dataset/gender_bias_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
total_f, total_steps = 0, 0
for i, (clean, corrupted, label) in enumerate(tqdm(dataloader)):
    single_data = [(clean, corrupted, label)]
    model.reset_hooks()
    
    g = Graph.from_model(model)

    print('evaluating baseline on this single data...')
    if i != 107:
        continue
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()

    # if baseline <= 1:
    #     continue
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    # print('attributing for this single data...')
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=steps, intervention=intervention, quiet=True)

    # print('evaluating circuit of this single data...')
    circuit_results = []
    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        g.prune()
        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False, a=True), hook_rep=True, hook_layer=True, hook_pattern=True, intervention=intervention, quiet=True)
        exit(0)
        results = results.mean().item()
        circuit_results.append(results)
        
        # faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
        circuit_faithfulness.append(faithfulness)

        print(f"Original performance: {baseline:.3f}; circuit performance: {results:.3f}; corrupted_baseline: {corrupted_baseline:.3f}; faithfulness: {faithfulness:.3f}")
        total_steps += 1
        total_f += faithfulness
        edges = {}
        for edge in g.edges.values():
            child_node_id = int(edge.child.name.split(".")[-1][1:]) if edge.child.name not in ['input', "logits"] else -1
            parent_node_id = int(edge.parent.name.split(".")[-1][1:]) if edge.parent.name not in ['input', "logits"] else -1
        
        if faithfulness > 0.9:
            # print('saving graph data at question idx ', i)
            meta_info = {'idx': i, 'topn': topn, 'faithfulness': faithfulness, 'baseline_bias_logit': baseline, 'clean': clean[0], 'corrupted': corrupted[0], 'clean_label': label.tolist()[0][0], 'corrupted_label': label.tolist()[0][1]}
            meta_info.update({'nodes_in_graph': g.nodes_in_graph.cpu().numpy().tolist()})
            meta_info.update({'in_graph': g.in_graph.cpu().numpy().tolist()})
            with open(f'graph_data/top{topn}/gender_bias_{meta_info["idx"]}_top{topn}.json', 'a') as f:
                json.dump(meta_info, f)
            
            g.to_json(f'graph_data/top{topn}/json/{meta_info["idx"]}_top{topn}.json')
        
    all_results.append({
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'topns': topns,
        'circuit_results': circuit_results,
        'circuit_faithfulness': circuit_faithfulness
    })

print(f'avg faithfulness: {total_f / total_steps:.3f} over {total_steps} steps')

# Calculate average faithfulness across all samples
one_sample_faithfulness = [d['circuit_faithfulness'] for d in all_results]
one_sample_faithfulness = np.mean(one_sample_faithfulness, axis=0)
one_sample_faithfulness = one_sample_faithfulness.tolist()

plt.plot(topns, one_sample_faithfulness, label=f'{method}', marker='o')

plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15, loc='lower right')
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
# plt.savefig(f'gender_bias_{data_num}_one_sample_faithfulness.png', dpi=500, bbox_inches='tight')