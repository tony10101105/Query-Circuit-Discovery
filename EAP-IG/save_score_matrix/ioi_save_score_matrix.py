import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed
from eap.query_circuit_utils import logit_diff, EAPDataset
from save_score_matrix.models import DatasetConfig, TargetModelConfig, DiscoveryAlgConfig
set_seed(2025)


dataset_cfg = DatasetConfig(num_samples=1000)
model_cfg = TargetModelConfig(model_name='gpt2-small')
alg_cfg = DiscoveryAlgConfig(
    topns=[50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000],  # 32491
)
score_matrix_save_dir = f'score_matrix/ioi/{model_cfg.model_name}'
os.makedirs(score_matrix_save_dir, exist_ok=True)

model = HookedTransformer.from_pretrained(model_cfg.model_name, device=model_cfg.device)
model.cfg.use_split_qkv_input = model_cfg.use_split_qkv_input
model.cfg.use_attn_result = model_cfg.use_attn_result
model.cfg.use_hook_mlp_in = model_cfg.use_hook_mlp_in
model.cfg.ungroup_grouped_query_attention = model_cfg.ungroup_grouped_query_attention

ds = EAPDataset('probing_dataset/ioi_gpt2.csv', num_samples=dataset_cfg.num_samples)
dataloader = ds.to_dataloader(batch_size=1)

all_results = []
for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing samples"):
    single_data = [(clean, corrupted, label)]
    
    model.reset_hooks()
    
    g = Graph.from_model(model)

    print('evaluating baseline on this single data...')
    baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), quiet=True).mean().item()
    corrupted_baseline = evaluate_baseline(model, single_data, partial(logit_diff, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()

    print('attributing for this single data...')
    attribute(model, g, single_data, partial(logit_diff, loss=True, mean=True), method=alg_cfg.method, ig_steps=alg_cfg.steps, intervention=alg_cfg.intervention, quiet=True)

    x = g.scores.cpu().detach().numpy()
    x[~g.real_edge_mask] = -np.inf
    np.save(f'{score_matrix_save_dir}/{i}.npy', x)