import random
import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
import numpy as np
from copy import deepcopy
import pandas as pd

from jaxtyping import Float
from functools import partial

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer, FactoredMatrix

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

set_seed(2025)

def show_induction_scores(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    fig = px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs
    )
    fig.write_image('induction_scores.png')
    
    
model_name = "gpt2-small"
torch.set_grad_enabled(False)
device = utils.get_device()
model = HookedTransformer.from_pretrained(model_name, device=device)

data_num = 500
seq_len = 20
size = (data_num, seq_len)
min_token, max_token = 1000, 10000
input_tensor = torch.randint(min_token, max_token, size)

random_tokens = input_tensor.to(model.cfg.device)
repeated_tokens = einops.repeat(random_tokens, "batch seq_len -> batch (2 seq_len)")

induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
def induction_score_hook(
    pattern: Float[torch.Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
    ):
    # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
    # This only has entries for tokens with index>=seq_len
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
    
    # delete first and last element, which are not used in the induction score
    induction_stripe = induction_stripe[:, :, 1:-1]
    
    # Get an average score per head
    induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score

# We make a boolean filter on activation names, that's true only on attention pattern names.
pattern_hook_names_filter = lambda name: name.endswith("pattern")

model.run_with_hooks(
    repeated_tokens, 
    return_type=None, # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(
        pattern_hook_names_filter,
        induction_score_hook
    )]
)

# show_induction_scores(induction_score_store, xaxis="Head", yaxis="Layer", title="Induction Score by Head")

induction_score_store = induction_score_store.cpu().numpy()

flat_indices = np.argpartition(induction_score_store.ravel(), -10)[-10:]

# Get the corresponding 2D indices
top_10_indices = np.array(np.unravel_index(flat_indices, induction_score_store.shape)).T

# Sort the top-10 indices by actual score (descending)
top_10_indices = top_10_indices[np.argsort(induction_score_store[tuple(top_10_indices.T)])[::-1]]
print(top_10_indices)


### generate the probing dataset 
# print('repeated_tokens ', repeated_tokens) # idx 50~99 is repeated tokens
repeated_tokens = repeated_tokens.cpu().numpy().tolist()

def trim_list(lst):
    assert len(lst) == seq_len*2, "Input list must have seq_len*2 elements"
    # target_length = random.randint(seq_len+1, 2*seq_len-1)
    target_length = 2*seq_len-1
    trimmed = lst[:target_length]
    next_token_gt = lst[target_length-seq_len]
    return trimmed, next_token_gt

def randomize_repeated_pattern(seq, final_token):
    pattern = seq[:seq_len]
    shuffled_pattern = deepcopy(pattern)
    random.shuffle(shuffled_pattern)
    shuffled_seq = shuffled_pattern + seq[seq_len:]

    replacer = random.choice([i for i in pattern if i != final_token])
    shuffled_seq = [replacer if token == final_token else token for token in shuffled_seq]
    shuffled_seq[-1] = final_token  # Ensure the last token is the same as the original
    return shuffled_seq

# def randomize_repeated_pattern(seq, final_token):
#     shuffled_pattern = deepcopy(seq[:-1])
#     random.shuffle(shuffled_pattern)
#     replacer = random.choice([i for i in shuffled_pattern if i != final_token])
#     shuffled_seq = [replacer if token == final_token else token for token in shuffled_pattern]
#     shuffled_seq += [final_token]  # Ensure the last token is the same as the original
#     return shuffled_seq

new_repeated_tokens, corrupted, gts, incorrects = [], [], [], [] # clean input, corrupted input, and label
for i in range(len(repeated_tokens)):
    trimmed_list, gt = trim_list(repeated_tokens[i])
    new_repeated_tokens.append(trimmed_list)

    randomized_list = randomize_repeated_pattern(trimmed_list, gt)
    
    corrupted.append(randomized_list)

    assert len(trimmed_list) == len(randomized_list), "Trimmed and randomized lists must have the same length"

    gts.append(gt) # correct tokens

    # generate incorrect tokens, which are tokens in the new_repeated_tokens but not the gt
    incorrect_idx = list(set([number for number in trimmed_list if number != gt]))
    incorrects.append(incorrect_idx)

df = pd.DataFrame({
    'clean': new_repeated_tokens,
    'corrupted': corrupted,
    'label': gts,
    'incorrect_label': incorrects
})

df.to_csv("probing_dataset/induction_gpt2.csv", index=False)