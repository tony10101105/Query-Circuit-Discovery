"""
# This script runs the EAP-IG method to find a canonical circuit for the induction task. We have to do this because there's no previous benchmarking induction task's canonical circuit.
"""

from functools import partial

import ast
import pygraphviz
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from transformer_lens import HookedTransformer

from src.eap.graph import Graph
from src.eap.evaluate import evaluate_graph, evaluate_baseline
from src.eap.attribute import attribute 

def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = list(labels)
    # labels = torch.tensor(labels)
    return clean, corrupted, labels

class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        return ast.literal_eval(row['clean']), ast.literal_eval(row['corrupted']), [int(row['label']), ast.literal_eval(row['incorrect_label'])]

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
    
def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)

    logits = logits[idx, input_length - 1]
    return logits

def logit_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    incorrect_label = [j for i, j in labels]
    # label to tensor
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)
    # print('before logits: ', logits.shape) # torch.Size([bs, n_pos, 50257])
    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample
    # print('after logits: ', logits.shape) # torch.Size([bs, 50257])
    
    # print(correct_label.shape) # torch.Size([bs])
    good = torch.gather(logits, -1, correct_label).squeeze(1)
    bad = []
    for i, sample_label in enumerate(incorrect_label):
        sample_label = torch.tensor(sample_label, device=logits.device)
        # print('sample_label: ', sample_label.shape) # torch.Size([n])
        sample_bad = torch.gather(logits[i], -1, sample_label)
        # print('sample_bad: ', sample_bad.shape)
        bad.append(sample_bad.mean())
    bad = torch.tensor(bad, device=logits.device)
    # print('bad: ', bad.shape) # torch.Size([bs])
    # good_bad = torch.gather(logits, -1, labels.to(logits.device))
    # print('good: ', good.shape)
    # results = good_bad[:, 0] - good_bad[:, 1]
    results = good - bad
    # print('results: ', results.shape) # torch.Size([bs])

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def prob_diff(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    incorrect_label = [j for i, j in labels]
    # label to tensor
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)
    # print('before logits: ', logits.shape) # torch.Size([bs, n_pos, 50257])
    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample
    probs = torch.softmax(logits, dim=-1)
    
    # print(correct_label.shape) # torch.Size([bs])
    good = torch.gather(probs, -1, correct_label).squeeze(1)
    bad = []
    for i, sample_label in enumerate(incorrect_label):
        sample_label = torch.tensor(sample_label, device=logits.device)
        sample_bad = torch.gather(probs[i], -1, sample_label)
        bad.append(sample_bad.mean())
    bad = torch.tensor(bad, device=logits.device)

    results = good - bad

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def log_prob(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    # label to tensor
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)
    # print('before logits: ', logits.shape) # torch.Size([bs, n_pos, 50257])
    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    good = torch.gather(log_probs, -1, correct_label).squeeze(1)
    results = good

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

def prob(logits: torch.Tensor, clean_logits: torch.Tensor, input_length: torch.Tensor, labels: torch.Tensor, mean=True, loss=False):
    correct_label = [i for i, j in labels]
    correct_label = torch.tensor(correct_label, device=logits.device).unsqueeze(1)

    logits = get_logit_positions(logits, input_length) # get the last token logits of each sample

    probs = torch.softmax(logits, dim=-1)

    good = torch.gather(probs, -1, correct_label).squeeze(1)
    results = good

    if loss:
        results = -results
    if mean: 
        results = results.mean()
    return results

topn = 32491 # 32491 is the number of all edges in gpt2-small's graph
model_name = 'gpt2-small'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True

ds = EAPDataset('probing_dataset/induction_gpt2.csv')
dataloader = ds.to_dataloader(batch_size=10)

g = Graph.from_model(model)

# Attribute using the model, graph, clean / corrupted data and labels, as well as a metric
print('attributing...')
attribute(model, g, dataloader, partial(log_prob, loss=True, mean=True), method='EAP-IG-inputs', ig_steps=5, induction=True)
# attribute(model, g, dataloader, partial(log_prob, loss=True, mean=True), method='information-flow-routes', induction=True)

g.apply_topn(topn, True)

print('left node number: ', g.count_included_nodes())
print('left edge number: ', g.count_included_edges())

# g.to_json(f'preprocessed_circuit/induction_{model_name}_top{topn}_edge_canonical_circuit.json')
g.to_json(f'preprocessed_circuit/gpt2.json')

# gz = g.to_image(f'induction_{model_name}_top{topn}_edge_canonical_circuit.png')

print('evaluating baseline...')
baseline = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False), induction=True).mean().item()

print('evaluating circuit...')
results, _, _, _ = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False), induction=True)
results = results.mean().item()

print(f"Original performance was {baseline}; the circuit's performance is {results}")