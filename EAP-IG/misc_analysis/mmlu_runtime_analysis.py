import os
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_root)
sys.path.insert(0, _root)

import argparse
from functools import partial
import time
from tqdm import tqdm
from transformer_lens import HookedTransformer

from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import set_seed, pad_corrupted_to_clean
from eap.query_circuit_utils import logit_diff, EAPDataset, PARAEAPDataset, ndf
set_seed(2025)


def parse_args():
    parser = argparse.ArgumentParser(description="MMLU runtime analysis (single or best-of-n)")
    parser.add_argument("--mode", choices=["single", "bon"], default="single",
                        help="'single': use only the original query; 'bon': pick best paraphrase per topn")
    parser.add_argument("--category", default="astronomy",
                        help="MMLU category (default: astronomy)")
    parser.add_argument("--rephrase_type", default="only_stem")
    parser.add_argument("--rephrase_model", default="gpt4o")
    parser.add_argument("--method", default="EAP-IG-inputs",
                        choices=["EAP-IG-inputs", "EAP-IG-activations"])
    parser.add_argument("--steps", type=int, default=20,
                        help="IG steps")
    parser.add_argument("--topns", type=int, nargs='+', default=[500, 2000, 5000, 10000, 30000, 50000, 100000, 150000, 200000, 250000, 300000])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B-Instruct")
    return parser.parse_args()


def run_single(model, dataloader, metric_fn, args):
    g = Graph.from_model(model)
    intervention = "zero" if args.method == "EAP-IG-activations" else "patching"
    all_results = []

    start = time.perf_counter()
    for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="single"):
        single_data = [([clean[j]], [corrupted[j]], [label[j]]) for j in range(len(clean))]

        baseline = evaluate_baseline(model, single_data, partial(metric_fn, loss=False, mean=False), quiet=True).mean().item()
        corrupted_baseline = evaluate_baseline(model, single_data, partial(metric_fn, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
        pad_corrupted_to_clean(model, single_data)

        attribute(model, g, single_data, partial(metric_fn, loss=True, mean=True),
                  method=args.method, ig_steps=args.steps, intervention=intervention,
                  quiet=True)

        circuit_faithfulness = []
        for topn in args.topns:
            g.apply_topn(topn, True)
            results, _, _, _ = evaluate_graph(model, g, single_data, partial(metric_fn, loss=False, mean=False),
                                              hook_rep=False, hook_layer=False, hook_pattern=False,
                                              intervention=intervention, quiet=True)
            results = results.mean().item()
            circuit_faithfulness.append(ndf(results, baseline, corrupted_baseline))

        all_results.append(circuit_faithfulness)

    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.2f} seconds")


def run_bon(model, dataloader, metric_fn, args):
    intervention = "zero" if args.method == "EAP-IG-activations" else "patching"
    all_best_results = []

    start = time.perf_counter()
    for i, (clean, corrupted, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="bon"):
        batch_slice_data = [([clean[j]], [corrupted[j]], [label[j]]) for j in range(len(clean))]
        batch_slice_data = batch_slice_data[:2]

        baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(metric_fn, loss=False, mean=False), quiet=True).mean().item()
        corrupted_baseline = evaluate_baseline(model, [batch_slice_data[0]], partial(metric_fn, loss=False, mean=False), run_corrupted=True, quiet=True).mean().item()
        
        pad_corrupted_to_clean(model, batch_slice_data)

        best_results = [-1] * len(args.topns)
        best_para_indices = [0] * len(args.topns)

        for j, single_data in enumerate(batch_slice_data):
            model.reset_hooks()
            g = Graph.from_model(model)

            attribute(model, g, [single_data], partial(metric_fn, loss=True, mean=True),
                      method=args.method, ig_steps=args.steps, intervention=intervention,
                      quiet=True)

            for idx, topn in enumerate(args.topns):
                g.apply_topn(topn, True)
                results, _, _, _ = evaluate_graph(model, g, [batch_slice_data[0]], partial(metric_fn, loss=False, mean=False),
                                                  hook_rep=False, hook_layer=False, hook_pattern=False,
                                                  intervention=intervention, quiet=True)
                results = results.mean().item()
                faithfulness = ndf(results, baseline, corrupted_baseline)
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

    metric_fn = partial(logit_diff, mc=True)
    if args.mode == "single":
        ds = EAPDataset(csv_path, num_samples=args.num_samples, mc=True)
        run_fn = run_single
    else:
        ds = PARAEAPDataset(csv_path, num_samples=args.num_samples)
        run_fn = run_bon

    dataloader = ds.to_dataloader(batch_size=1)
    print(f"Mode: {args.mode} | Samples: {len(dataloader)} | Steps: {args.steps}")

    run_fn(model, dataloader, metric_fn, args)