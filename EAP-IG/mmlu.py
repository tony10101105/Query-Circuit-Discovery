from functools import partial

import ast
import json
import matplotlib.pyplot as plt
import pandas as pd
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
    labels = list(labels)
    # labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath, category=None, num_samples=None):
        self.df = pd.read_csv(filepath)
        if category:
            self.df = self.df[self.df['category'] == category]
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples)
        print(f'Loaded {len(self.df)} samples from {filepath} with category {category} and {len(self.df)} samples')

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return row['clean'], row['corrupted'], [int(row['correct_idx']), ast.literal_eval(row['incorrect_idx'])]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: list[list],             # list of [correct_idx, [wrong_idx1, wrong_idx2, ...]]
    mean: bool = True,
    loss: bool = False,
):
    # Extract the last-token logits: [batch_size, vocab_size]
    logits = get_logit_positions(logits, input_length)
    probs = torch.softmax(logits, dim=-1)

    batch_size = logits.size(0)

    # Get correct token logits
    correct_idxs = torch.tensor([lbl[0] for lbl in labels], device=logits.device)
    correct_logits = logits[torch.arange(batch_size), correct_idxs]

    # (4) correct - average over all wrong option tokens
    bad_vals = []
    for i, (_, wrong_ids) in enumerate(labels):
        wrong_logits = logits[i, torch.tensor(wrong_ids, device=logits.device)]
        bad_vals.append(wrong_logits.mean())
    bad = torch.stack(bad_vals)

    results = correct_logits - bad

    if loss:
        results = -results
    if mean:
        results = results.mean()
    return results



topns = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000] # 386713 for llama
category = 'astronomy' # marketing, professional_medicine, astronomy, college_biology, high_school_computer_science, logical_fallacies, nutrition, international_law, management
method = 'EAP-IG-inputs' # EAP-IG-inputs # EAP-IG-activations
steps = 20
intervention = 'zero' if method == 'EAP-IG-activations' else 'patching'
model_name = 'meta-llama/Llama-3.2-1B-Instruct' # gpt2-small # meta-llama/Llama-3.2-1B # meta-llama/Meta-Llama-3-8B-Instruct
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
model.cfg.ungroup_grouped_query_attention = True

ds = EAPDataset(f'probing_dataset/mmlu_{category}_Llama-32-1B.csv', num_samples=500)
dataloader = ds.to_dataloader(batch_size=1)

g = Graph.from_model(model)

print('evaluating baseline...')
baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
corrupted_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), run_corrupted=True).mean().item()
print(f"Original performance was {baseline}; the corrupted_baseline performance is {corrupted_baseline}")

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
print('attributing...')
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method=method, ig_steps=5, intervention=intervention)

print('evaluating circuit...')
circuit_results = []
circuit_faithfulness = []
for topn in topns:
    g.apply_topn(topn, True)

    print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

    # g.to_json(f'ioi_{model_name}_{topn}_circuit.json')
    # gz = g.to_image(f'ioi_{model_name}_{topn}_circuit.png')

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=intervention)
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

exit(0)
plt.plot(topns, circuit_faithfulness, label=method, marker='o')  # marker adds dots on points

plt.ylim(-0.1, 1.1)
plt.xlim(0, max(topns)+200)
plt.xlabel('Top-K Edges')
plt.ylabel('Circuit Faithfulness')
plt.title('IOI Circuit Faithfulness vs Top-K Edges')
plt.legend()

plt.savefig('ioi_circuit_faithfulness.pdf', bbox_inches='tight')
