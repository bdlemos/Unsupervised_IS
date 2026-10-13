"""
Graph Diffusion & Submodular Facility Location Instance Selection (DiffusionRank-IS).

Features:
1. Cosine k-NN Graph & Random-Walk Transition Matrix.
2. Stationary distribution / PageRank-style manifold diffusion centrality (pi).
3. Fast Lazy-Greedy Facility Location with density weighting.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import scipy.sparse as sp


def compute_diffusion_weights(X, k=25, alpha=0.85):
    """
    Computes manifold diffusion centrality using a k-NN transition matrix.
    alpha is the teleportation parameter (standard PageRank style diffusion).
    """
    n = len(X)
    k = min(k, n - 1)
    
    # 1. Cosine k-NN
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute', n_jobs=4).fit(X)
    dist, indices = nn.kneighbors(X)
    
    # Cosine distance to similarity: S = 1 - dist = cos(x, y)
    sim = np.maximum(0.0, 1.0 - dist[:, 1:])  # exclude self
    neighbors = indices[:, 1:]
    
    # Adaptive local bandwidth sigma_i: median distance to k-NN
    sigmas = np.median(dist[:, 1:], axis=1, keepdims=True) + 1e-8
    weights = np.exp(- (dist[:, 1:] ** 2) / (sigmas ** 2))
    
    # Row normalize to form stochastic transition matrix P
    row_sums = weights.sum(axis=1, keepdims=True) + 1e-12
    norm_weights = weights / row_sums
    
    # Sparse Transition Matrix
    rows = np.repeat(np.arange(n), k)
    cols = neighbors.flatten()
    data = norm_weights.flatten()
    P = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Power iteration for PageRank / Stationary diffusion distribution
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, 1.0 / n, dtype=np.float64)
    
    for _ in range(30):
        pi_next = alpha * P.T.dot(pi) + (1.0 - alpha) * teleport
        if np.linalg.norm(pi_next - pi, 1) < 1e-6:
            break
        pi = pi_next
        
    return pi


import heapq


def lazy_greedy_facility_location(X, weights, n_keep, seed=0):
    """
    Selects n_keep instances by maximizing density-weighted facility location:
        F(S) = sum_i max_{j in S} (sim(x_i, x_j) * weights[j])
    Uses Minoux (1978) lazy greedy with priority queue and target subsampling for scalability.
    """
    n = len(X)
    if n_keep >= n:
        return np.arange(n)
        
    rng = np.random.RandomState(seed)
    # Scale weights so mean is 1.0
    w = (weights / (np.mean(weights) + 1e-12)).astype(np.float32)
    
    # Candidate pool: top candidates by diffusion weight + small random exploration
    pool_size = min(n, max(int(n_keep * 2.5), 800))
    top_candidates = np.argpartition(-w, pool_size - 1)[:pool_size]
    cand_X = X[top_candidates].astype(np.float32)
    cand_w = w[top_candidates]
    
    # Target set for coverage evaluation: if N is huge, evaluate on representative sample of size min(n, 3000)
    if n > 3000:
        target_idx = rng.choice(n, size=3000, replace=False)
        target_X = X[target_idx].astype(np.float32)
    else:
        target_X = X.astype(np.float32)
        
    n_targets = len(target_X)
    max_sim = np.zeros(n_targets, dtype=np.float32)
    # Exact initial upper bounds on targets via single fast matrix multiplication
    sim_init = np.maximum(0.0, cand_X.dot(target_X.T))
    init_gains = sim_init.sum(axis=1) * cand_w
    
    # Minoux Lazy Greedy Priority Queue: (-upper_bound, candidate_pool_idx)
    heap = [(-float(g), i) for i, g in enumerate(init_gains)]
    heapq.heapify(heap)
    
    selected_pool_indices = []
    
    while len(selected_pool_indices) < n_keep and heap:
        neg_gain, best_c = heapq.heappop(heap)
        
        # Recompute exact marginal gain against all targets
        sim_c = np.maximum(0.0, target_X.dot(cand_X[best_c])) * cand_w[best_c]
        true_gain = float(np.maximum(0.0, sim_c - max_sim).sum())
        
        # Submodularity check: if true_gain >= upper bound of next element in heap, it's optimal
        next_best = -heap[0][0] if heap else 0.0
        if true_gain >= next_best or not heap:
            selected_pool_indices.append(best_c)
            max_sim = np.maximum(max_sim, sim_c)
        else:
            heapq.heappush(heap, (-true_gain, best_c))
            
    selected = top_candidates[selected_pool_indices]
    return np.sort(np.array(selected, dtype=int))


def diffusion_cover(X, weights, n_keep, seed=0, gamma=0.5, k_max=1000):
    """
    Selects n_keep instances by sublinear cluster tessellation and 
    prioritizing manifold diffusion hubs (pi) within each cluster.
    """
    from sklearn.cluster import MiniBatchKMeans
    from proxy_lib import _quotas
    n = len(X)
    if n_keep >= n:
        return np.arange(n)
        
    K = int(min(n_keep, k_max, n // 2))
    km = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=3, batch_size=4096)
    labels = km.fit_predict(X)
    sizes = np.bincount(labels, minlength=K)
    q = _quotas(sizes, gamma, n_keep)
    
    keep = []
    for c in range(K):
        idx = np.where(labels == c)[0]
        if len(idx) == 0 or q[c] == 0:
            continue
        # Pick top instances by diffusion centrality pi
        keep.append(idx[np.argsort(-weights[idx])[:q[c]]])
    return np.sort(np.concatenate(keep))

