import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, PARAEAPDataset
from save_score_matrix.models import DatasetConfig, TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


dataset_cfg = DatasetConfig(num_samples=500)
model_cfg = TargetModelConfig(model_name='meta-llama/Llama-3.2-1B-Instruct')
alg_cfg = DiscoveryAlgConfig(
    topns=[500, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 30000, 40000, 50000],
)
score_matrix_save_dir = 'score_matrix/arithmetic_add/llama32-1b'
os.makedirs(score_matrix_save_dir, exist_ok=True)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

ds = PARAEAPDataset('probing_dataset/arithmetic_add_Llama-32-1B.csv', num_samples=dataset_cfg.num_samples, simple=True)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in enumerate(tqdm(dataloader, total=len(dataloader), desc="Processing samples")):
    assert len(clean) == len(corrupted) and len(corrupted) == len(label)
    batch_slice_data = [([clean[0][j]], [corrupted[0][j]], torch.tensor([label[0][j]])) for j in range(len(clean[0]))]

    model.reset_hooks()
    g = Graph.from_model(model)

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False)).mean().item()
    corrupted_baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(logit_diff, loss=False, mean=False), run_corrupted=True).mean().item()
    pad_corrupted_to_clean(model, batch_slice_data)

    print('attributing for this single data and save the score matrix...')
    attribute(model, g, batch_slice_data, partial(logit_diff, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention, score_matrix_save_dir=score_matrix_save_dir, file_idx=i)
