"""k50: separation without labels. The data-only mid-range attraction (PaCMAP-style mid-near pairs, weight 0.1 in
the SGD pass) halves the gap between classes on MNIST (median within-class spread / median centroid distance 0.22
against 0.12 without it; k-means ARI 0.74 against 0.93). Variants, each on MNIST 70k, two seeds, SGD pass:
  default        mid weight 0.1
  mid0           mid weight 0
  mid_in_comm    mid weight 0.1, mid-near pairs that cross a graph community (the post-process partition: Louvain
                 at resolution 0.5 + interface merge, label-free) redirected to the point's strongest neighbour
  soft_graph f   mid weight 0.1, the graph edges between communities multiplied by f (0.5, 0.082 = exp(-2.5))
Axes: 15-NN preservation, distance Spearman (benchmarks/metrics.py evaluate), k-means ARI (k = 10, 10 inits), the
spread / gap ratio. Also: the post-process (contraction 0.7) on the default and on the soft graph, and a mid-range phase cut to the
first quarter of the epochs (mid_phase 0.25). Usage: python benchmarks/theory/k50_community_soft.py [loss=umap]"""
import os, sys, json, time, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("NUMBA_NUM_THREADS", "8")
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B); sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from datasets import mnist
from metrics import evaluate
from magnet import MAGNET
from magnet.contraction import communities, interface_merge
T = 8
kw = dict(a.split("=", 1) for a in sys.argv[1:]); LOSS = kw.get("loss", "umap")
OUT = os.path.join(HERE, "k50_community_soft.json"); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
X, y, _ = mnist(70000)

def partition(P):
    return interface_merge(P, communities(P, 0.5))

class MidInComm(MAGNET):
    def _label_mid_pairs(self, pairs, P):
        if pairs is None: return pairs
        self._y_fit_ = partition(P); self.n_comm_ = int(self._y_fit_.max() + 1)
        try: return super()._label_mid_pairs(pairs, P)
        finally: self._y_fit_ = None

class SoftGraph(MAGNET):
    def __init__(self, factor=0.5, **k):
        super().__init__(**k); self.factor = factor
    def build_graph(self, X):
        P = super().build_graph(X); c = partition(P); self.n_comm_ = int(c.max() + 1)
        C = P.tocoo(); f = np.where(c[C.row] == c[C.col], 1.0, self.factor).astype(np.float32)
        Q = sp.csr_matrix((C.data * f, (C.row, C.col)), shape=P.shape); Q.sort_indices(); return Q

def gap_ratio(Y):
    C = np.array([Y[y == c].mean(0) for c in range(10)])
    w = np.median([np.median(np.linalg.norm(Y[y == c] - C[c], axis=1)) for c in range(10)])
    D = np.linalg.norm(C[:, None] - C[None], axis=2); return float(w / np.median(D[np.triu_indices(10, 1)]))

V = {"default": lambda s: MAGNET(loss=LOSS, random_state=s, n_threads=T),
     "mid0": lambda s: MAGNET(loss=LOSS, mid_weight=0.0, random_state=s, n_threads=T),
     "mid_in_comm": lambda s: MidInComm(loss=LOSS, random_state=s, n_threads=T),
     "soft_graph 0.5": lambda s: SoftGraph(0.5, loss=LOSS, random_state=s, n_threads=T),
     "soft_graph 0.082": lambda s: SoftGraph(0.082, loss=LOSS, random_state=s, n_threads=T),
     "default + contraction": lambda s: MAGNET(loss=LOSS, contraction=0.7, random_state=s, n_threads=T),
     "mid_phase 0.25": lambda s: MAGNET(loss=LOSS, mid_phase=0.25, random_state=s, n_threads=T),
     "soft_graph 0.082 + contraction": lambda s: SoftGraph(0.082, loss=LOSS, contraction=0.7, random_state=s, n_threads=T)}
for name, mk in V.items():
    for s in range(2):
        key = f"{LOSS} | {name} | {s}"
        if key in res: continue
        m = mk(s); t0 = time.time(); Y = m.fit_transform(X); t = time.time() - t0
        e = evaluate(X, Y, labels=y)
        r = dict(nn15=e["knn15_preservation"], spearman=e["dist_spearman"], knn_acc=e.get("knn_acc"),
                 ari=float(adjusted_rand_score(y, KMeans(10, n_init=10, random_state=0).fit_predict(Y))),
                 spread_gap=gap_ratio(Y), n_comm=getattr(m, "n_comm_", None), fit_s=round(t, 1))
        res[key] = r; print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
