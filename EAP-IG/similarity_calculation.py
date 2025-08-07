import torch
import numpy as np
import torch.nn.functional as F
import cca_core
from sklearn.decomposition import PCA
from CKA import linear_CKA, kernel_CKA
import pwcca


def get_representation_similarity(embeddings_X, embeddings_Y, methods=['cca', 'svcca', 'pwcca', 'linear cka', 'rbf cka']):
    """
    Calculate the similarity matrix between two sets of embeddings using the specified method.
    
    Args:
        embeddings_X (np.ndarray. (feature, sample)): Embeddings from the first model.
        embeddings_Y (np.ndarray. (feature, sample)): Embeddings from the second model.
        method (list): List of methods to use for calculating similarity ('svcca', 'cca', 'pwcca', 'cka').

    Returns:
        np.ndarray: Similarity matrix.
    """
    results = {}
    for method in methods:
        if method == 'cca':
            result = cca_core.get_cca_similarity(embeddings_X, embeddings_Y, epsilon=1e-10, verbose=False)
            results[method] = 1-np.mean(result["cca_coef1"][:20])

        elif method == 'svcca':
            cacts1 = embeddings_X - np.mean(embeddings_X, axis=1, keepdims=True)
            cacts2 = embeddings_Y - np.mean(embeddings_Y, axis=1, keepdims=True)
            U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
            U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)
            svacts1 = np.dot(s1[:20]*np.eye(20), V1[:20])
            # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
            svacts2 = np.dot(s2[:20]*np.eye(20), V2[:20])
            # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

            svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
            results[method] = 1-np.mean(svcca_results["cca_coef1"])
        
        elif method == 'pwcca':
            pwcca_mean, w, _ = pwcca.compute_pwcca(embeddings_X, embeddings_Y, epsilon=1e-10)
            results[method] = 1-pwcca_mean
        
        elif method == 'linear cka':
            result = linear_CKA(embeddings_X.T, embeddings_Y.T)
            results[method] = 1-result
        
        elif method == 'rbf cka':
            result = kernel_CKA(embeddings_X.T, embeddings_Y.T)
            results[method] = 1-result
        else:
            raise ValueError("Method must be either 'svcca', 'cca', 'pwcca', or 'cka'.")

    return results


# def get_js_div(X, Y):
#     """
#     A stupid but safe method to calculate the average JS divergence of two sets of probability distributions.
    
#     Args:
#         X (torch.Tensor): shape (batch, n_pos, n_pos), probs from model 1.
#         Y (torch.Tensor): shape (batch, n_pos, n_pos), probs from model 2.

#     Returns:
#         float: JS divergence
#     """
#     assert X.shape == Y.shape
#     cnt, total_js_div = 0, 0
#     eps = 1e-10
#     sample_size, token_num = X.shape[0], X.shape[1]
#     for i in range(sample_size):
#         for j in range(token_num):
#             if sum(X[i, j, :]) != 0 and sum(Y[i, j, :]) != 0:
#                 cnt += 1
#                 total_m = 0.5 * (X[i, j, :] + Y[i, j, :])
#                 loss = 0.0
#                 # print('X: ', X[i, j, :])
#                 # print('Y: ', Y[i, j, :])
#                 # exit(0)
#                 loss += F.kl_div(torch.log(X[i, j, :]+eps), total_m, reduction="batchmean") 
#                 loss += F.kl_div(torch.log(Y[i, j, :]+eps), total_m, reduction="batchmean") 
#                 loss *= 0.5
#                 total_js_div += loss
#             elif (sum(X[i, j, :]) != 0 and sum(Y[i, j, :]) == 0) or (sum(X[i, j, :]) == 0 and sum(Y[i, j, :]) != 0):
#                 raise Exception('you should not have this cuz X and Y are corresponding')
#             else:
#                 pass # padded ones
#     avg_js_div = total_js_div / cnt
#     return avg_js_div

def get_js_div(X, Y, eps=1e-10):
    """
    Vectorized JS divergence for two sets of probability distributions.

    Args:
        X (torch.Tensor): shape (batch, n_pos, n_pos), probs from model 1.
        Y (torch.Tensor): shape (batch, n_pos, n_pos), probs from model 2.

    Returns:
        float: average JS divergence over valid positions.
    """
    assert X.shape == Y.shape

    # Compute masks for valid rows (non-zero rows in both X and Y)
    valid_mask = ((X.sum(dim=-1) != 0) & (Y.sum(dim=-1) != 0))  # shape: (batch, n_pos)

    # # Normalize to ensure proper probability distributions (row-wise)
    # X_norm = X / (X.sum(dim=-1, keepdim=True) + eps)
    # Y_norm = Y / (Y.sum(dim=-1, keepdim=True) + eps)

    M = 0.5 * (X + Y)

    # Compute JS divergence: 0.5 * (KL(X || M) + KL(Y || M))    
    kl_xm = F.kl_div(torch.log(X + eps), M, reduction='none').sum(dim=-1)
    kl_ym = F.kl_div(torch.log(Y + eps), M, reduction='none').sum(dim=-1)
    js = 0.5 * (kl_xm + kl_ym)
    
    # Apply valid mask and compute average
    js_valid = js[valid_mask]
    avg_js = js_valid.mean().item()

    return avg_js