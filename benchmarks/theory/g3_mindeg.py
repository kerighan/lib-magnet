"""Minimum degree for the adaptive switch on graph inputs: chains (degree 2) benefit from L2, leaves (degree 1) do not."""
import sys, os, numpy as np, scipy.sparse as sp
REPO = sys.argv[1]; sys.path.insert(0, REPO)
from magnet import MAGNET
from sklearn.neighbors import NearestNeighbors
import networkx as nx
def edge_pres_fn(A):
    A = sp.csr_matrix(A); deg = np.diff(A.indptr); k = int(min(50, deg.max()))
    def f(Y):
        lnn = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
        out = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            if deg[i]: out[i] = len(np.intersect1d(A.indices[A.indptr[i]:A.indptr[i + 1]], lnn[i, :deg[i]])) / deg[i]
        return out
    return f
RW = "/home/maixent/Reputation Squad Dropbox/Maixent Chenebaux/DATAPROJECTS/lib/rolewalk/benchmarks/data"
G = nx.read_edgelist(os.path.join(RW, "usa-airports.edgelist"), nodetype=int); lab = {}
for line in open(os.path.join(RW, "labels-usa-airports.txt")):
    p = line.split()
    if p[0] != "node": lab[int(p[0])] = int(p[1])
nodes = sorted(G.nodes()); A_air = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=nodes)).astype(np.float32); g_air = np.array([lab.get(v, -1) for v in nodes])
sizes = [300] * 5; p = [[0.002] * 5 for _ in range(5)]
for i in range(5): p[i][i] = 0.05
G = nx.stochastic_block_model(sizes, p, seed=0); n0 = G.number_of_nodes()
for b in range(5):
    start = n0 + 300 * b; nx.add_path(G, range(start, start + 300)); G.add_edge(sum(sizes[:b]), start)
g_sbm = np.array(["block"] * n0 + ["chain"] * 1500); A_sbm = sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()))).astype(np.float32)
for name, A, groups in (("usa airports", A_air, g_air), ("SBM + chains", A_sbm, g_sbm)):
    metric = edge_pres_fn(A); deg = np.diff(sp.csr_matrix(A).indptr); gl = np.unique(groups)
    print(f"== {name}: degree quantiles 10/50/90 = {np.percentile(deg,10):.0f}/{np.median(deg):.0f}/{np.percentile(deg,90):.0f}; frac deg<=2 {np.mean(deg<=2):.2f}")
    confs = [("const 0.5", dict(kernel_b=0.5)), ("const 0.6", dict(kernel_b=0.6))] + [(f"adapt .5@.75 md{md}", dict(kernel_b="adaptive", adaptive=(0.5, 1.0, 0.75, 0.0, 2.0, md))) for md in (1, 2, 3, 5)] + [(f"adapt .6@.75 md{md}", dict(kernel_b="adaptive", adaptive=(0.6, 1.0, 0.75, 0.0, 2.0, md))) for md in (2, 3)]
    for label, c in confs:
        sc, fr = [], []
        for seed in (0, 1, 2):
            M = MAGNET(n_threads=4, random_state=seed, **c).fit(A); sc.append(metric(M.embedding_))
            if M.node_b_ is not None: fr.append(M.node_b_ == 1)
        e = np.mean(sc, 0); extra = ("  | b=1 " + " ".join(f"{g}:{np.mean(fr,0)[groups==g].mean():.2f}" for g in gl[:7]) + f"  deg<=2 switched {np.mean(fr,0)[deg<=2].mean():.2f}") if fr else ""
        print(f"  {label:18s} score {e.mean():.4f}  " + " ".join(f"{g}:{e[groups==g].mean():.3f}" for g in gl[:7]) + extra, flush=True)
