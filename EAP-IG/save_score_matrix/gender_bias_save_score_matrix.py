import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed
from eap.query_circuit_utils import logit_diff, EAPDataset
from save_score_matrix.models import TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--score_matrix_save_dir', type=str, default='score_matrix/gender_bias/gpt2-small')
parser.add_argument('--dataset_path', type=str, default='probing_dataset/gender_bias_gpt2.csv')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig(topns=[100], steps=5)
os.makedirs(args.score_matrix_save_dir, exist_ok=True)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

ds = EAPDataset(args.dataset_path,
                correct_col='clean_answer_idx', incorrect_col='corrupted_answer_idx')
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
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention, score_matrix_save_dir=args.score_matrix_save_dir, file_idx=i, quiet=True)

    # print('evaluating circuit of this single data...')
    circuit_results = []
    circuit_faithfulness = []
    for topn in alg_cfg.topns:
        g.apply_topn(topn, True)
        g.prune()
        print(f'top{topn}. Node, edge number: {g.count_included_nodes()}, {g.count_included_edges()}')

        results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True, intervention=alg_cfg.intervention, quiet=True)
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
        'topns': alg_cfg.topns,
        'circuit_results': circuit_results,
        'circuit_faithfulness': circuit_faithfulness
    })

print(f'avg faithfulness: {total_f / total_steps:.3f} over {total_steps} steps')

# Calculate average faithfulness across all samples
one_sample_faithfulness = [d['circuit_faithfulness'] for d in all_results]
one_sample_faithfulness = np.mean(one_sample_faithfulness, axis=0)
one_sample_faithfulness = one_sample_faithfulness.tolist()

plt.plot(alg_cfg.topns, one_sample_faithfulness, label=f'{alg_cfg.method}', marker='o')

plt.ylim(-0.1, 1.1)
plt.xlabel('Number of Edges', fontsize=15)
plt.ylabel('Normalized Deviation Faithfulness (NDF)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15, loc='lower right')
plt.grid(True, which='both', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
# plt.savefig(f'gender_bias_{data_num}_one_sample_faithfulness.png', dpi=500, bbox_inches='tight')