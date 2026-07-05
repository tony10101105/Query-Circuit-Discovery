import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
import json

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.utils import set_seed
from eap.sae_utils import load_sae_suite, run_with_sae_feature_ablation
from save_score_matrix.models import TargetModelConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2-small')
parser.add_argument('--biased_samples_path', type=str, default='query_circuit_behavior_analysis/data/biased_samples.json')
parser.add_argument('--paired_summary_path', type=str, default='query_circuit_behavior_analysis/data/paired_summary.json')
parser.add_argument('--features_dir', type=str, default='query_circuit_behavior_analysis/data/circuit_features')
parser.add_argument('--output_path', type=str, default='query_circuit_behavior_analysis/data/paired_steering_results.json')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.eval()

sae_suite = load_sae_suite(range(model.cfg.n_layers), model_cfg.device)

with open(args.biased_samples_path, 'r') as f:
    biased_samples = {s['idx']: s for s in json.load(f)}
with open(args.paired_summary_path, 'r') as f:
    paired_summary = json.load(f)


def measure_bias(sample, ablate_features):
    last_logits = run_with_sae_feature_ablation(model, sae_suite, sample['clean'], ablate_features)
    probs = torch.softmax(last_logits, dim=-1)
    logit_bias = (last_logits[sample['clean_answer_idx']] - last_logits[sample['corrupted_answer_idx']]).item()
    prob_bias = (probs[sample['clean_answer_idx']] - probs[sample['corrupted_answer_idx']]).item()
    return logit_bias, prob_bias


def steer_circuit(sample, which):
    with open(os.path.join(args.features_dir, f'paired_target{sample["idx"]}_{which}.json'), 'r') as f:
        features = json.load(f)
    ablate_features = {k: v for k, v in features['gender_features'].items() if k.endswith('-mlp')}
    num_features = sum(len(v) for v in ablate_features.values())

    logit_before, prob_before = measure_bias(sample, {})
    logit_after, prob_after = measure_bias(sample, ablate_features)

    sign = 1.0 if logit_before > 0 else -1.0
    logit_reduction = sign * (logit_before - logit_after)
    prob_reduction = sign * (prob_before - prob_after)

    return {
        'source_idx': features['source_idx'],
        'num_gender_features_mlp': num_features,
        'logit_bias_before': logit_before,
        'prob_bias_before': prob_before,
        'logit_bias_after': logit_after,
        'prob_bias_after': prob_after,
        'logit_reduction': logit_reduction,
        'prob_reduction': prob_reduction,
        'logit_reduction_per_feature': logit_reduction / num_features if num_features > 0 else 0.0,
        'prob_reduction_per_feature': prob_reduction / num_features if num_features > 0 else 0.0,
    }


results = []
for pair in tqdm(paired_summary):
    if not pair['selected']:
        continue
    sample = biased_samples[pair['idx']]
    results.append({
        'idx': pair['idx'],
        'best': steer_circuit(sample, 'best'),
        'worst': steer_circuit(sample, 'worst'),
    })

metrics = [
    ('Absolute Bias Reduction', 'Logit', 'logit_reduction'),
    ('Absolute Bias Reduction', 'Probability', 'prob_reduction'),
    ('Avg. Bias Reduction per Gender Feature', 'Logit', 'logit_reduction_per_feature'),
    ('Avg. Bias Reduction per Gender Feature', 'Probability', 'prob_reduction_per_feature'),
]

table = []
for metric_name, scale, key in metrics:
    best_vals = np.array([r['best'][key] for r in results])
    worst_vals = np.array([r['worst'][key] for r in results])
    diffs = best_vals - worst_vals

    two_sided = stats.wilcoxon(diffs)
    one_sided = stats.wilcoxon(diffs, alternative='greater', method='approx')
    rosenthal_r = abs(one_sided.zstatistic) / np.sqrt(len(diffs))

    row = {
        'metric': metric_name,
        'scale': scale,
        'best_mean': best_vals.mean(),
        'best_std': best_vals.std(),
        'worst_mean': worst_vals.mean(),
        'worst_std': worst_vals.std(),
        'delta_mean': diffs.mean(),
        'W': float(two_sided.statistic),
        'p_value': float(one_sided.pvalue),
        'rosenthal_r': float(rosenthal_r),
    }
    table.append(row)

with open(args.output_path, 'w') as f:
    json.dump({'num_pairs': len(results), 'table': table, 'per_sample': results}, f, indent=2)

baseline_probs = np.abs([biased_samples[r['idx']]['prob_bias'] for r in results])
print(f'\n{len(results)} pairs; baseline probability bias: {baseline_probs.mean():.3f} ± {baseline_probs.std():.3f}\n')
print(f'{"Metric":<42}{"Scale":<14}{"Best mean±std":<18}{"Worst mean±std":<18}{"ΔMean":<10}{"W":<8}{"p-value":<12}{"r":<8}')
for row in table:
    print(f'{row["metric"]:<42}{row["scale"]:<14}'
          f'{row["best_mean"]:.3f} ± {row["best_std"]:.3f}   '
          f'{row["worst_mean"]:.3f} ± {row["worst_std"]:.3f}   '
          f'{row["delta_mean"]:+.3f}   {row["W"]:<8.1f}{row["p_value"]:<12.6f}{row["rosenthal_r"]:.3f}')
print(f'\nresults saved to {args.output_path}')
