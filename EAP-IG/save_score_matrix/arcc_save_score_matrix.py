import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial

from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, PARAEAPDataset
from save_score_matrix.models import DatasetConfig, TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


parser = argparse.ArgumentParser()
parser.add_argument('--category', type=str, default='challenge')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
parser.add_argument('--num_samples', type=int, default=-1)
parser.add_argument('--score_matrix_save_dir', type=str, default='score_matrix/arc_challenge/llama32-1b')
parser.add_argument('--dataset_path', type=str, default=None)
args = parser.parse_args()

dataset_cfg = DatasetConfig(category=args.category, num_samples=args.num_samples)
model_cfg = TargetModelConfig(model_name=args.model_name)
alg_cfg = DiscoveryAlgConfig()
score_matrix_save_dir = args.score_matrix_save_dir
os.makedirs(score_matrix_save_dir, exist_ok=True)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

dataset_path = args.dataset_path or f'probing_dataset/arc_{dataset_cfg.category}_Llama-32-1B_{dataset_cfg.rephrase_model}_paraphrases_{dataset_cfg.rephrase_type}.csv'
ds = PARAEAPDataset(dataset_path)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    assert len(clean) == len(corrupted) and len(corrupted) == len(label)
    
    batch_slice_data = [([clean[j]], [corrupted[j]], [label[j]]) for j in range(len(clean))]
    model.reset_hooks()
    g = Graph.from_model(model)
    
    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, mc=True, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, mc=True, loss=False, mean=False), run_corrupted=True).mean().item()

    pad_corrupted_to_clean(model, batch_slice_data)

    print('attributing for this single data and save the score matrix...')
    attribute(model, g, batch_slice_data, partial(logit_diff, mc=True, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention, score_matrix_save_dir=score_matrix_save_dir, file_idx=i)