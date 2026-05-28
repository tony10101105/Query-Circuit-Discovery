"""
Shared utilities developed for our query circuit scripts.

Exports:
  get_logit_positions  -- extract last-token logits from a batch
  logit_diff           -- unified logit diff (mc=False: gather-based; mc=True: correct − mean wrong)
  collate_EAP          -- collate for EAP samples (tensor_labels=True/False)
  collate_para         -- collate for PARA datasets (batch_size=1 paraphrase expansion)
  EAPDataset           -- unified flat-CSV dataset (mc=False/True controls label parsing)
  PARAEAPDataset       -- unified 10-paraphrase dataset (simple=False/True)
"""

import ast
from functools import partial
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(logits.size(0), device=logits.device)
    return logits[idx, input_length - 1]


def logit_diff(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels,
    mc: bool = False,
    mean: bool = True,
    loss: bool = False,
) -> torch.Tensor:
    """Unified logit diff.

    mc=False (default): gather-based for simple 2-token tensor labels [correct, incorrect].
    mc=True: correct logit − mean(wrong logits) for MC list labels [[correct, [wrong...]], ...].
    """
    logits = get_logit_positions(logits, input_length)
    if mc:
        batch_size = logits.size(0)
        correct_idxs = torch.tensor([lbl[0] for lbl in labels], device=logits.device)
        correct_logits = logits[torch.arange(batch_size), correct_idxs]
        bad = torch.stack([
            logits[i, torch.tensor(wrong, device=logits.device)].mean()
            for i, (_, wrong) in enumerate(labels)
        ])
        results = correct_logits - bad
    else:
        good_bad = torch.gather(logits, -1, labels.to(logits.device))
        results = good_bad[:, 0] - good_bad[:, 1]
    
    if loss:
        results = -results
    if mean:
        results = results.mean()
    
    return results





def collate_EAP(xs, tensor_labels: bool = True):
    """Collate EAP samples.

    tensor_labels=True (default): stacks labels into a tensor (simple 2-token integer labels).
    tensor_labels=False: keeps labels as a Python list (MC or nested-list labels).
    """
    clean, corrupted, labels = zip(*xs)
    return list(clean), list(corrupted), torch.tensor(labels) if tensor_labels else list(labels)


def collate_para(xs):
    """Collate for PARA datasets with batch_size=1 paraphrase expansion."""
    clean, corrupted, labels = zip(*xs)
    return clean[0], corrupted[0], labels[0]


class EAPDataset(Dataset):
    """Unified flat-CSV EAP dataset.

    mc=False (default): simple [correct_idx, incorrect_idx] labels; incorrect_idx kept as-is.
    mc=True: MC labels [correct_idx, [wrong_idx, ...]]; incorrect_idx parsed with ast.literal_eval.

    Args:
        filepath: path to CSV file
        data_num: keep first N rows (mutually exclusive with slice_start/slice_end)
        num_samples: alias for data_num (takes precedence)
        category: filter rows by this value in the 'category' column
        slice_start / slice_end: iloc range
        clean_col, corrupted_col, correct_col, incorrect_col: column name overrides
        mc: if True, parse incorrect_idx as a list (MC label format)
    """

    def __init__(
        self,
        filepath,
        data_num=None,
        num_samples=None,
        category=None,
        slice_start=None,
        slice_end=None,
        clean_col="clean",
        corrupted_col="corrupted",
        correct_col="correct_idx",
        incorrect_col="incorrect_idx",
        mc=False,
    ):
        self.df = pd.read_csv(filepath)
        self.clean_col = clean_col
        self.corrupted_col = corrupted_col
        self.correct_col = correct_col
        self.incorrect_col = incorrect_col
        self.mc = mc
        if category:
            self.df = self.df[self.df["category"] == category]
        effective_num = num_samples if num_samples is not None else data_num
        if effective_num is not None and effective_num < len(self.df):
            self.df = self.df.head(effective_num)
        elif slice_start is not None or slice_end is not None:
            self.df = self.df.iloc[slice_start:slice_end]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        correct = int(row[self.correct_col]) if self.mc else row[self.correct_col]
        incorrect = (
            ast.literal_eval(row[self.incorrect_col]) if self.mc else row[self.incorrect_col]
        )
        return row[self.clean_col], row[self.corrupted_col], [correct, incorrect]

    def to_dataloader(self, batch_size=1):
        collate = partial(collate_EAP, tensor_labels=False) if self.mc else collate_EAP
        return DataLoader(self, batch_size=batch_size, collate_fn=collate)


class PARAEAPDataset(Dataset):
    """Unified 10-paraphrase EAP dataset.

    simple=False (default): MC labels — incorrect_idx parsed with ast.literal_eval; uses collate_para.
    simple=True: simple integer incorrect_idx; uses tensor_labels=False collate (arithmetic paraphrase files).

    Expects columns: clean, corrupted, paraphrase1..paraphrase9, correct_idx, incorrect_idx.
    """

    def __init__(self, filepath, num_samples=None, data_num=None, simple=False):
        self.df = pd.read_csv(filepath)
        self.simple = simple
        n = num_samples if num_samples is not None else data_num
        if n is not None:
            self.df = self.df.head(n)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        clean = [row["clean"]] + [row[f"paraphrase{i}"] for i in range(1, 10)]
        corrupted = [row["corrupted"]] * 10
        correct_idx = int(row["correct_idx"])
        incorrect_idx = (
            int(row["incorrect_idx"]) if self.simple else ast.literal_eval(row["incorrect_idx"])
        )
        labels = [[correct_idx, incorrect_idx]] * 10
        return clean, corrupted, labels

    def to_dataloader(self, batch_size=1):
        # simple=True (arithmetic): tensor_labels=False keeps outer list → callers use clean[0][j]
        # simple=False (MC): collate_para strips outer list → callers use clean[j]
        collate = partial(collate_EAP, tensor_labels=False) if self.simple else collate_para
        return DataLoader(self, batch_size=batch_size, collate_fn=collate)
