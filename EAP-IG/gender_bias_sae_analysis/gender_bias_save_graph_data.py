import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial
import json
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, EAPDataset, ndf
from save_score_matrix.models import TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--topn', type=int, default=100)
parser.add_argument('--faithfulness_threshold', type=float, default=0.9)
parser.add_argument('--baseline_threshold', type=float, default=1.0)
parser.add_argument('--dataset_path', type=str, default='probing_dataset/gender_bias_gpt2.csv')
parser.add_argument('--output_dir', type=str, default='gender_bias_sae_analysis/graph_data')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig()

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

ds = EAPDataset(args.dataset_path, correct_col='clean_answer_idx', incorrect_col='corrupted_answer_idx')
dataloader = ds.to_dataloader(batch_size=1)

out_dir = os.path.join(args.output_dir, f'top{args.topn}')
os.makedirs(os.path.join(out_dir, 'json'), exist_ok=True)

for i, (clean, corrupted, label) in enumerate(tqdm(dataloader)):
    single_data = [(clean, corrupted, label)]
    model.reset_hooks()

    g = Graph.from_model(model)

    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()

    if baseline <= args.baseline_threshold: # if model on this sample is not biased enough, skip it
        continue

    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    pad_corrupted_to_clean(model, single_data)
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention, quiet=True)

    g.apply_topn(args.topn, True)
    g.prune()

    results, _, _, _ = evaluate_graph(model, g, single_data, partial(logit_diff, loss=False, mean=False), hook_rep=True, hook_layer=True, hook_pattern=True, intervention=alg_cfg.intervention, quiet=True)
    results = results.mean().item()

    faithfulness = ndf(results, baseline, corrupted_baseline)

    if faithfulness > args.faithfulness_threshold: # if the sample's discovered circuit is faithful enough, save the graph data
        print(f'saving graph data at question idx {i}')
        meta_info = {
            'idx': i,
            'topn': args.topn,
            'faithfulness': faithfulness,
            'baseline_bias_logit': baseline,
            'clean': clean[0],
            'corrupted': corrupted[0],
            'clean_label': label.tolist()[0][0],
            'corrupted_label': label.tolist()[0][1],
            'nodes_in_graph': g.nodes_in_graph.cpu().numpy().tolist(),
            'in_graph': g.in_graph.cpu().numpy().tolist(),
        }
        with open(os.path.join(out_dir, f'gender_bias_{i}_top{args.topn}.json'), 'a') as f:
            json.dump(meta_info, f)

        g.to_json(os.path.join(out_dir, 'json', f'{i}_top{args.topn}.json'))
