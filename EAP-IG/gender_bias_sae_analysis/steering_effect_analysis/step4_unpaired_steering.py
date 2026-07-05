import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
parser.add_argument('--biased_samples_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/biased_samples.json')
parser.add_argument('--unpaired_summary_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/unpaired_summary.json')
parser.add_argument('--features_dir', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/circuit_features')
parser.add_argument('--output_path', type=str, default='gender_bias_sae_analysis/steering_effect_analysis/data/unpaired_steering_results.json')
args = parser.parse_args()

model_cfg = TargetModelConfig(model_name=args.model_name)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.eval()

sae_suite = load_sae_suite(range(model.cfg.n_layers), model_cfg.device)

with open(args.biased_samples_path, 'r') as f:
    biased_samples = {s['idx']: s for s in json.load(f)}
with open(args.unpaired_summary_path, 'r') as f:
    unpaired_summary = json.load(f)


def measure_bias(sample, ablate_features):
    last_logits = run_with_sae_feature_ablation(model, sae_suite, sample['clean'], ablate_features)
    probs = torch.softmax(last_logits, dim=-1)
    logit_bias = (last_logits[sample['clean_answer_idx']] - last_logits[sample['corrupted_answer_idx']]).item()
    prob_bias = (probs[sample['clean_answer_idx']] - probs[sample['corrupted_answer_idx']]).item()
    return logit_bias, prob_bias


results = []
for entry in tqdm(unpaired_summary):
    sample = biased_samples[entry['idx']]
    with open(os.path.join(args.features_dir, f'unpaired_target{entry["idx"]}.json'), 'r') as f:
        features = json.load(f)
    ablate_features = {k: v for k, v in features['gender_features'].items() if k.endswith('-mlp')}
    num_features = sum(len(v) for v in ablate_features.values())

    logit_before, prob_before = measure_bias(sample, {})
    logit_after, prob_after = measure_bias(sample, ablate_features)

    sign = 1.0 if logit_before > 0 else -1.0
    logit_reduction = sign * (logit_before - logit_after)
    prob_reduction = sign * (prob_before - prob_after)

    results.append({
        'idx': entry['idx'],
        'group': entry['group'],
        'ndf': entry['ndf'],
        'num_gender_features_mlp': num_features,
        'logit_bias_before': sign * logit_before,
        'prob_bias_before': sign * prob_before,
        'logit_reduction': logit_reduction,
        'prob_reduction': prob_reduction,
        'logit_reduction_per_feature': logit_reduction / num_features if num_features > 0 else 0.0,
        'prob_reduction_per_feature': prob_reduction / num_features if num_features > 0 else 0.0,
    })

metrics = [
    ('Bias Before Steer', 'Probability', 'prob_bias_before'),
    ('Bias Before Steer', 'Logit', 'logit_bias_before'),
    ('Bias Reduction', 'Probability', 'prob_reduction'),
    ('Bias Reduction', 'Logit', 'logit_reduction'),
    ('Avg. Bias Reduction per Gender Feature', 'Probability', 'prob_reduction_per_feature'),
    ('Avg. Bias Reduction per Gender Feature', 'Logit', 'logit_reduction_per_feature'),
]

table = []
for metric_name, scale, key in metrics:
    high_vals = np.array([r[key] for r in results if r['group'] == 'high'])
    low_vals = np.array([r[key] for r in results if r['group'] == 'low'])

    test = stats.mannwhitneyu(high_vals, low_vals, alternative='greater')
    pooled_std = np.sqrt(((len(high_vals) - 1) * high_vals.std(ddof=1) ** 2 + (len(low_vals) - 1) * low_vals.std(ddof=1) ** 2) / (len(high_vals) + len(low_vals) - 2))
    cohens_d = (high_vals.mean() - low_vals.mean()) / pooled_std

    row = {
        'metric': metric_name,
        'scale': scale,
        'high_mean': high_vals.mean(),
        'high_std': high_vals.std(),
        'low_mean': low_vals.mean(),
        'low_std': low_vals.std(),
        'high_n': len(high_vals),
        'low_n': len(low_vals),
        'delta_mean': high_vals.mean() - low_vals.mean(),
        'U': float(test.statistic),
        'p_value': float(test.pvalue),
        'cohens_d': float(cohens_d),
    }
    table.append(row)

with open(args.output_path, 'w') as f:
    json.dump({'table': table, 'per_sample': results}, f, indent=2)

print(f'\nhigh-NDF group: {table[0]["high_n"]} samples, low-NDF group: {table[0]["low_n"]} samples\n')
print(f'{"Metric":<42}{"Scale":<14}{"High mean±std":<18}{"Low mean±std":<18}{"ΔMean":<10}{"U":<8}{"p-value":<12}{"d":<8}')
for row in table:
    print(f'{row["metric"]:<42}{row["scale"]:<14}'
          f'{row["high_mean"]:.3f} ± {row["high_std"]:.3f}   '
          f'{row["low_mean"]:.3f} ± {row["low_std"]:.3f}   '
          f'{row["delta_mean"]:+.3f}   {row["U"]:<8.1f}{row["p_value"]:<12.6f}{row["cohens_d"]:.3f}')
print(f'\nresults saved to {args.output_path}')
