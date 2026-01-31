import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering, DBSCAN

from sklearn.preprocessing import LabelEncoder
from scipy.optimize import linear_sum_assignment

from sklearn.neighbors import NearestNeighbors

def hierarchical_clustering(D, all_head_idx_tup, all_head_idx, all_head_2_class):
    condensed_D = squareform(D, checks=False)
    # 2. Compute the linkage matrix
    Z = linkage(condensed_D, method='average')  # you can also try 'complete', 'ward', 'single', etc.

    # 3. (Optional) Assign flat clusters, e.g., 5 clusters:
    num_clusters = 5
    labels = fcluster(Z, t=num_clusters, criterion='maxclust')  # labels.shape == (144,)
    
    plt.figure(figsize=(10, 6))
    node_names = [all_head_2_class[idx] for idx in all_head_idx_tup]
    dendrogram(
        Z,
        labels=node_names,
        truncate_mode='level',  # show only the top levels
        p=30,                   # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=8.
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Node index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('hierarchical_clustering.png')
    return 0

def spectral_clustering(D, all_head_idx_tup, all_head_idx, all_head_2_class):
    node_names = [all_head_2_class[idx] for idx in all_head_idx_tup]
    # 1. Perform spectral clustering
    
    gamma = 1.0 / D.std()  # you may tune this
    gammas = np.linspace(0, 2, 21)
    for gamma in gammas:
        print(f"gamma: {gamma}")
        Dd = np.exp(-gamma * D ** 2)
        # d_min, d_max = D.min(), D.max()
        # D_norm = (D - d_min) / (d_max - d_min)
        # D = 1.0 - D_norm

        num_clusters = 7
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=2025)
        labels = clustering.fit_predict(Dd)
        labels = np.array(labels, dtype=int)

        le = LabelEncoder()
        y_true = le.fit_transform(node_names)
        # Now strings → ints; you can recover strings via le.inverse_transform

        n_clusters = len(np.unique(labels))
        n_classes  = len(le.classes_)

        # 3) Build the contingency table (clusters × true_classes)
        w = np.zeros((n_clusters, n_classes), dtype=int)
        for c,p in zip(labels, y_true):
            w[c, p] += 1

        # 4) Solve the assignment problem (maximize matches)
        #    Hungarian finds the minimal cost; cost = max(w) - w
        row_ind, col_ind = linear_sum_assignment(w.max() - w)

        # 5) Compute accuracy
        total_correct = w[row_ind, col_ind].sum()
        accuracy = total_correct / len(y_true)

        print(f"Cluster→Class mapping:")
        for cluster_id, class_id in zip(row_ind, col_ind):
            print(f"  cluster {cluster_id} → {le.inverse_transform([class_id])[0]}")
        print(f"\nClustering accuracy = {accuracy:.3f} ({total_correct}/{len(y_true)})")
    exit(0)

    # 2. Plot the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(all_head_idx, np.zeros_like(all_head_idx), c=labels, cmap='viridis', s=50)
    plt.xticks(all_head_idx, node_names, rotation=90)
    plt.title('Spectral Clustering of Nodes')
    plt.xlabel('Node Index')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.savefig('spectral_clustering.png')
    return labels


def dbscan_clustering(D, all_head_idx_tup, all_head_idx, all_head_2_class):
    node_names = [all_head_2_class[idx] for idx in all_head_idx_tup]
    D[D < 0] = 0
    db = DBSCAN(metric="precomputed", eps=0.001, min_samples=1)
    labels = db.fit_predict(D)
    labels = np.array(labels, dtype=int)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)

    print(f"Found {n_clusters} clusters")
    print(f"Found {n_noise} noise points")
    print(f"Labels: {labels}")
    
    # nbrs = NearestNeighbors(n_neighbors=2, metric="precomputed").fit(D)
    # distances, _ = nbrs.kneighbors(D)
    # k_dist = np.sort(distances[:, 1])  # 2nd neighbor

    # plt.figure(figsize=(10, 6))
    # plt.plot(k_dist)
    # plt.ylabel("4th NN distance")
    # plt.xlabel("Points sorted by distance")
    # plt.title("k-distance plot for DBSCAN eps selection")
    # plt.savefig('dbscan_k_distance_plot.png')
    
    # 2) Encode string labels to integers 0…C−1
    le = LabelEncoder()
    y_true = le.fit_transform(node_names)

    max_cluster = labels.max()
    n_clusters = max_cluster + 1
    
    # Build contingency matrix W of shape (n_clusters × C)
    n_classes = len(le.classes_)
    W = np.zeros((n_clusters, n_classes), dtype=int)
    for pred_cluster, true_cls in zip(labels, y_true):
        if pred_cluster >= 0:  # skip noise if any
            W[pred_cluster, true_cls] += 1
    
    # Majority‐vote mapping: each cluster → the class with highest count in W
    mv_mapping = {cluster: W[cluster].argmax() for cluster in range(n_clusters)}
    # Optionally map noise (–1) to a dummy class unreachable in y_true:
    mv_mapping[-1] = -1
    
    # Apply mapping
    y_matched = np.array([ mv_mapping[c] for c in labels ])
    
    # Compute accuracy (ignore noise points or count them as incorrect)
    accuracy = np.mean(y_matched == y_true)
    
    # Print mapping and result
    print("Majority‐vote cluster→class mapping:")
    for cluster, cls_id in mv_mapping.items():
        cls_name = le.inverse_transform([cls_id])[0] if cls_id >= 0 else "noise"
        print(f"  cluster {cluster} → {cls_name}")
    print(f"\nAccuracy = {accuracy:.3f} ({int((y_matched == y_true).sum())}/{len(y_true)})")
    
    return labels