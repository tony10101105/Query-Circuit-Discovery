import math
import numpy as np

import numpy as np
import os

topns = [50, 100, 250, 500, 1000, 1500, 2000]

def jaccard_similarity_from_masks(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    sim = intersection / union
    return sim

for topn in topns:
    f1 = f"EAP-IG-inputs_{topn}_connected_edges_mask.npy"
    f2 = f"EAP-IG-inputs-sg_{topn}_connected_edges_mask.npy"

    if not os.path.exists(f1):
        print(f"[{topn}] missing file: {f1}")
        continue
    if not os.path.exists(f2):
        print(f"[{topn}] missing file: {f2}")
        continue

    a = np.load(f1)
    b = np.load(f2)
    dist = jaccard_similarity_from_masks(a, b)
    print(f"top{topn}: Jaccard similarity = {dist:.4f}")
