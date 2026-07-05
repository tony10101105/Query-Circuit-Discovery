import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.query_circuit_utils import logit_diff, EAPDataset, nfs
from save_score_matrix.models import TargetModelConfig, DiscoveryAlgConfig


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--topns', type=int, nargs='+', default=[50, 100, 200, 300, 400, 500, 750, 1000])
parser.add_argument('--dataset_path', type=str, default='probing_dataset/gender_bias_gpt2.csv')
parser.add_argument('--output_figure', type=str, default='figures/gender_bias_topn_vs_dijkstra.pdf')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig()

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

ds = EAPDataset(args.dataset_path, correct_col='clean_answer_idx', incorrect_col='corrupted_answer_idx')
dataloader = ds.to_dataloader(batch_size=10)

g = Graph.from_model(model)

print('evaluating baseline...')
baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()
corrupted_baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), run_corrupted=True).mean().item()
print(f'Baseline: {baseline}, Corrupted Baseline: {corrupted_baseline}')

print('attributing...')
attribute(model, g, dataloader, partial(logit_diff, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention)
print('evaluating circuit...')
circuit_faithfulness_g, circuit_faithfulness_t = [], []
for topn in tqdm(args.topns):
    g.apply_topn(topn, True)
    g.prune()

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention)
    results = results.mean().item()

    circuit_faithfulness_t.append(round(nfs(results, baseline, corrupted_baseline), 2))

    g.apply_greedy(topn, True)
    g.prune()

    results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False), hook_rep=False, hook_layer=False, hook_pattern=False, intervention=alg_cfg.intervention)
    results = results.mean().item()

    circuit_faithfulness_g.append(round(nfs(results, baseline, corrupted_baseline), 2))

print('topns: ', args.topns)
print('circuit_faithfulness_t: ', circuit_faithfulness_t)
print('circuit_faithfulness_g: ', circuit_faithfulness_g)
plt.plot(args.topns, circuit_faithfulness_t, label='Greedy Selection', marker='o')
plt.plot(args.topns, circuit_faithfulness_g, label='Dijkstra-like Construction', marker='o')

plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=16)
plt.ylabel('Normalized Faithfulness Score (NFS)', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, loc='lower right')
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(args.output_figure, bbox_inches='tight')