"""Hub hypothesis for the text optimum (0.6-0.7 instead of 0.5): 20 newsgroups' contamination is bimodal, a fifth of
the edges end far beyond the kernel scale (hubs no kernel can satisfy). If hubs are what moves the optimum, damping
them on the graph side should move it back to 0.5. Variants of the neighbour graph, all fed to the optimiser as a
graph (same init, same walk-based mid-range pairs): the fuzzy union (baseline), the fuzzy intersection (mutual
neighbours), a degree cap on hubs, and the symmetric degree normalisation. MNIST 20k as the control."""
import sys, os, time, numpy as np, scipy.sparse as sp
REPO = sys.argv[1]; sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "benchmarks"))
from magnet import MAGNET
from datasets import mnist, news20
from metrics import evaluate

def cap_degree(P, cap):
    P = sp.csr_matrix(P); n = P.shape[0]; keep = sp.lil_matrix((n, n), dtype=bool)
    deg = np.diff(P.indptr)
    for i in range(n):
        s, e = P.indptr[i], P.indptr[i + 1]
        if deg[i] <= cap: keep[i, P.indices[s:e]] = True
        else:
            top = P.indices[s:e][np.argsort(-P.data[s:e])[:cap]]; keep[i, top] = True
    keep = keep.tocsr(); keep = keep.multiply(keep.T)        # mutual within the cap
    return P.multiply(keep).tocsr()

def band(P0, Y, dk):
    i = np.repeat(np.arange(P0.shape[0]), np.diff(P0.indptr)); d = np.sqrt(((Y[i] - Y[P0.indices]) ** 2).sum(1))
    return np.mean(d > dk), np.mean((d > dk) & (d <= 2)), np.mean(d > 2), d

for name, (X, y, key), kw in (("news20", news20(), {"metric": "cosine"}), ("mnist20k", mnist(20000), {})):
    base = MAGNET(n_threads=4, random_state=0, **kw); P0 = base.build_graph(X).tocsr()
    ind = base.knn_indices_; indeg = np.bincount(ind.ravel(), minlength=len(X))
    top = np.argsort(-indeg)[:len(X) // 100]
    print(f"== {name}: directed kNN in-degree max {indeg.max()} p99 {np.percentile(indeg,99):.0f} median {np.median(indeg):.0f}; share of in-edges to the top 1% nodes {indeg[top].sum()/ind.size:.3f}; fuzzy-union degree max {np.diff(P0.indptr).max()} p99 {np.percentile(np.diff(P0.indptr),99):.0f}", flush=True)
    Pint = MAGNET(n_threads=4, random_state=0, set_op_mix_ratio=0.0, **kw).build_graph(X).tocsr()
    variants = [("union (baseline)", P0, None), ("intersection", Pint, None), ("degree cap 30", cap_degree(P0, 30), None), ("sym degree norm", P0, "sym")]
    for label, P, gn in variants:
        print(f"-- {label}: edges {P.nnz}, degree p50/p99/max {np.median(np.diff(P.indptr)):.0f}/{np.percentile(np.diff(P.indptr),99):.0f}/{np.diff(P.indptr).max()}", flush=True)
        for b in (0.4, 0.5, 0.6, 0.7, 0.8):
            sc, bands, hub_far = [], [], []
            for seed in (0, 1):
                M = MAGNET(kernel_b=b, input="graph", graph_norm=gn, mid_source="walk", n_threads=4, random_state=seed).fit(P)
                r = evaluate(X, M.embedding_, y, cache_key=key); sc.append([r["knn15_preservation"], r["trustworthiness"], r["dist_spearman"], r["knn_acc"]])
                dk = M.a_ ** (-1 / (2 * M.b_)); p_far, p_band, p_very, d = band(P0, M.embedding_, dk); bands.append([p_far, p_band, p_very])
                j = P0.indices; hub_far.append(indeg[j][d > 2].mean() / max(indeg[j].mean(), 1e-9))
            s = np.mean(sc, 0); bb = np.mean(bands, 0)
            print(f"   b={b:<4} knn15 {s[0]:.4f} trust {s[1]:.4f} spear {s[2]:.4f} acc {s[3]:.4f} | original edges: beyond dk {bb[0]:.3f} band dk..2 {bb[1]:.3f} beyond 2 {bb[2]:.3f} | in-degree of far endpoints / mean {np.mean(hub_far):.2f}", flush=True)
