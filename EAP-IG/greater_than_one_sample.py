from functools import partial

import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute 


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        # self.df = self.df[:1]

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], row['label']
    
    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def get_prob_diff(tokenizer: PreTrainedTokenizer):
    year_indices = torch.tensor([tokenizer(f'{year:02d}').input_ids[0] for year in range(100)])

    def prob_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
        logits = get_logit_positions(logits, input_length)
        probs = torch.softmax(logits, dim=-1)[:, year_indices]

        results = []
        for prob, year in zip(probs, labels):
            results.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
    
        results = torch.stack(results)
        if loss:
            results = -results
        if mean: 
            results = results.mean()
        return results
    return prob_diff

def kl_div(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=True):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    results = kl_div(probs.log(), clean_probs.log(), log_target=True, reduction='none').mean(-1)
    return results.mean() if mean else results


topns = [50, 100, 200, 300, 400, 500, 750, 1000] # 32491
method = 'EAP-IG-activations' # 'EAP-IG-inputs'
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('probing_dataset/greater_than_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=1)
prob_diff = get_prob_diff(model.tokenizer)

all_results = []
for clean, corrupted, label in tqdm(dataloader):
    single_data = [(clean, corrupted, label)]
    model.reset_hooks()
    
    g = Graph.from_model(model)

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(prob_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(prob_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    print('attributing for this single data...')
    attribute(model, g, single_data, partial(prob_diff, loss=True, mean=True), method=method, ig_steps=5, intervention=intervention, quiet=True)

    print('evaluating circuit of this single data...')
    circuit_results = []
    circuit_faithfulness = []
    for topn in topns:
        g.apply_topn(topn, True)
        g.prune()
        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, single_data, partial(prob_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True, intervention=intervention, quiet=True)
        results = results.mean().item()
        circuit_results.append(results)
        
        faithfulness = (results - corrupted_baseline) / (baseline - corrupted_baseline)
        circuit_faithfulness.append(faithfulness)
        
        print(f"Original performance was {baseline}; the circuit's performance is {results}; faithfulness is {faithfulness}")

    all_results.append({
        'baseline': baseline,
        'corrupted_baseline': corrupted_baseline,
        'topns': topns,
        'circuit_results': circuit_results,
        'circuit_faithfulness': circuit_faithfulness
    })

with open(f'gt_{method.lower()}_one_sample_data.json', 'w') as f:
    json.dump(all_results, f, indent=2)

"""
plt.plot(topns, circuit_faithfulness, label=method, marker='o')  # marker adds dots on points

plt.ylim(-0.1, 1.1)
plt.xlim(0, max(topns)+200)
plt.xlabel('Top-K Edges')
plt.ylabel('Circuit Faithfulness')
plt.title('GT Circuit Faithfulness vs Top-K Edges')
plt.legend()

plt.savefig('oseap_figures/gt_circuit_faithfulness.png')
"""
    
"""
# Instantiate a graph with a model
g_eap = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_eap, dataloader, partial(prob_diff, loss=True, mean=True), method='EAP')

g_eap.apply_topn(200, True)
# print(f'g1 node number: {g_eap.count_included_nodes()}, edge number: {g_eap.count_included_edges()}')
# exit(0)
results_eap, _, _, _ = evaluate_graph(model, g_eap, dataloader, partial(prob_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True)
results_eap = results_eap.mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_eap}")


# Instantiate a graph with a model
g_cc = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
attribute(model, g_cc, dataloader, partial(prob_diff, loss=True, mean=True), method='clean-corrupted')

g_cc.apply_topn(200, True)

results_cc, _, _, _ = evaluate_graph(model, g_cc, dataloader, partial(prob_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True)
results_cc = results_cc.mean().item()
print(f"Original performance was {baseline}; the circuit's performance is {results_cc}")
"""