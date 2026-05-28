import os as _os; _os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from functools import partial
import os
import sys
import re
import ast
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import topn_indices, set_seed
set_seed(2025)


def parse_args():
    parser = argparse.ArgumentParser(description="MMLU runtime analysis (single or best-of-n)")
    parser.add_argument("--mode", choices=["single", "bon"], default="single",
                        help="'single': use only the original query; 'bon': pick best paraphrase per topn")
    parser.add_argument("--category", default="astronomy",
                        help="MMLU category (default: astronomy)")
    parser.add_argument("--rephrase_type", default="only_stem")
    parser.add_argument("--rephrase_model", default="gpt4o", choices=["gpt4o", "gpt4o-mini"])
    parser.add_argument("--method", default="EAP-IG-inputs",
                        choices=["EAP-IG-inputs", "EAP-IG-activations"])
    parser.add_argument("--steps", type=int, default=None,
                        help="IG steps (default: 5 for single, 20 for bon)")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    return parser.parse_args()


def _collate(xs):
    clean, corrupted, labels = zip(*xs)
    return clean[0], corrupted[0], labels[0]

class SingleEAPDataset(Dataset):
    """One (clean, corrupted) pair per row — no paraphrases."""

    def __init__(self, filepath, num_samples=None):
        self.df = pd.read_csv(filepath)
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples)
        print(f"Loaded {len(self.df)} samples from {filepath}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        clean = [row["clean"]]
        corrupted = [row["corrupted"]]
        correct_idx = int(row["correct_idx"])
        incorrect_idx = ast.literal_eval(row["incorrect_idx"])
        labels = [[correct_idx, incorrect_idx]]
        return clean, corrupted, labels

    def to_dataloader(self, batch_size=1):
        return DataLoader(self, batch_size=batch_size, collate_fn=_collate)


class BoNEAPDataset(Dataset):
    """Ten (clean, corrupted) pairs per row — original + 9 paraphrases."""

    def __init__(self, filepath, num_samples=None):
        self.df = pd.read_csv(filepath)
        if num_samples and num_samples < len(self.df):
            self.df = self.df.head(num_samples)
        print(f"Loaded {len(self.df)} samples from {filepath}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        clean = [row["clean"]] + [row[f"paraphrase{i}"] for i in range(1, 10)]
        corrupted = [row["corrupted"]] * 10
        correct_idx = int(row["correct_idx"])
        incorrect_idx = ast.literal_eval(row["incorrect_idx"])
        labels = [[correct_idx, incorrect_idx]] * 10
        return clean, corrupted, labels

    def to_dataloader(self, batch_size=1):
        return DataLoader(self, batch_size=batch_size, collate_fn=_collate)


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length - 1]


def logit_diff_logit(logits, clean_logits, input_length, labels, mean=True, loss=False):
    """Logit-based metric: correct token logit minus mean wrong token logits."""
    logits = get_logit_positions(logits, input_length)
    batch_size = logits.size(0)
    correct_idxs = torch.tensor([lbl[0] for lbl in labels], device=logits.device)
    correct_logits = logits[torch.arange(batch_size), correct_idxs]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TOPNS = [500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000]


def run_single(model, dataloader, metric_fn, args):
    g = Graph.from_model(model)
    intervention = "zero" if args.method == "EAP-IG-activations" else "patching"
    all_results = []

    start = time.perf_counter()
    for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="single"):
        single_data = [([clean[j]], [corrupted[j]], [label[j]]) for j in range(len(clean))]

        baseline = evaluate_baseline(model, single_data, partial(metric_fn, loss=False, mean=False), quiet=True).mean().item()
        corrupted_baseline = evaluate_baseline(model, single_data, partial(metric_fn, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
        _ = evaluate_baseline(model, single_data, partial(metric_fn, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

        attribute(model, g, single_data, partial(metric_fn, loss=True, mean=True),
                  method=args.method, ig_steps=args.steps, intervention=intervention,
                  file_idx=i, cat=args.category, quiet=True)

        circuit_faithfulness = []
        for topn in TOPNS:
            g.apply_topn(topn, True)
            results, _, _, _ = evaluate_graph(model, g, single_data, partial(metric_fn, loss=False, mean=False),
                                              hook_rep=False, hook_layer=False, hook_pattern=False,
                                              intervention=intervention, quiet=True)
            results = results.mean().item()
            faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
            circuit_faithfulness.append(faithfulness)

        all_results.append(circuit_faithfulness)

    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.2f} seconds")

    all_results = np.array(all_results).mean(0)
    print("all_results:", all_results)
    plt.plot(TOPNS, all_results, label="Single Query", marker="o")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Number of Edges (k)", fontsize=15)
    plt.ylabel("Normalized Deviation Faithfulness (NDF)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc="lower right", fontsize=15)
    plt.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    out = f"single_runtime_{args.steps}.png"
    plt.savefig(out)
    print(f"Saved plot to {out}")


def run_bon(model, dataloader, metric_fn, args):
    intervention = "zero" if args.method == "EAP-IG-activations" else "patching"
    all_best_results = []

    start = time.perf_counter()
    for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="bon"):
        batch_slice_data = [([clean[j]], [corrupted[j]], [label[j]]) for j in range(len(clean))]
        batch_slice_data = batch_slice_data[:2]

        baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(metric_fn, loss=False, mean=False), quiet=True).mean().item()
        corrupted_baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(metric_fn, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
        _ = evaluate_baseline(model, batch_slice_data, partial(metric_fn, loss=False, mean=False), run_corrupted=True, manual_pad=True, quiet=True)

        best_results = [-1] * len(TOPNS)
        best_para_indices = [0] * len(TOPNS)

        for j, single_data in enumerate(batch_slice_data):
            model.reset_hooks()
            g = Graph.from_model(model)

            attribute(model, g, [single_data], partial(metric_fn, loss=True, mean=True),
                      method=args.method, ig_steps=args.steps, intervention=intervention,
                      file_idx=i, cat=args.category, quiet=True)

            for idx, topn in enumerate(TOPNS):
                g.apply_topn(topn, True)
                results, _, _, _ = evaluate_graph(model, g, [batch_slice_data[0]], partial(metric_fn, loss=False, mean=False),
                                                  hook_rep=False, hook_layer=False, hook_pattern=False,
                                                  intervention=intervention, quiet=True)
                results = results.mean().item()
                faithfulness = 1 - min(abs((baseline - results) / (baseline - corrupted_baseline)), 1)
                if faithfulness > best_results[idx]:
                    best_results[idx] = faithfulness
                    best_para_indices[idx] = j

        all_best_results.append(best_results)

    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    args = parse_args()

    model = HookedTransformer.from_pretrained(args.model_name, device="cuda")
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    csv_path = (
        f"probing_dataset/mmlu_{args.category}_Llama-32-1B_"
        f"{args.rephrase_model}_paraphrases_{args.rephrase_type}.csv"
    )

    metric_fn = logit_diff_logit
    if args.mode == "single":
        ds = SingleEAPDataset(csv_path, num_samples=args.num_samples)
        run_fn = run_single
    else:
        ds = BoNEAPDataset(csv_path, num_samples=args.num_samples)
        run_fn = run_bon

    dataloader = ds.to_dataloader(batch_size=1)
    print(f"Mode: {args.mode} | Samples: {len(dataloader)} | Steps: {args.steps}")

    run_fn(model, dataloader, metric_fn, args)
