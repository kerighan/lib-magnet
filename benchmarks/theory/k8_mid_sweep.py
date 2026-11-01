"""k8: parametrisation of loss="hybrid" on MNIST 70k. One SGD pass per setting (mid_weight, epochs, seed), then the
exact pass with one knob moved at a time (scale, learning rate, iterations, dof, exaggeration). Usage:
python benchmarks/theory/k8_mid_sweep.py OUT.jsonl [round]  (round 1: exact-pass knobs; 2: mid_weight x epochs x seed).
Run from the repository root. Results in k8_mid_sweep.log."""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np
from datasets import LOADERS
from metrics import evaluate
from magnet import MAGNET
from magnet.tsne import optimize_tsne

data, labels, name = LOADERS["mnist"](70000)
n = len(data)
out = open(sys.argv[1], "a")

def log(row):
    print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()

def exact(P, Y0, scale=31.0, lr=None, iters=300, dof=1.0, exag=1.0):
    Y = Y0.copy(); Y -= Y.mean(0); Y *= scale * np.sqrt(n / 70000.0) / Y[:, 0].std()
    t = time.time()
    Y = optimize_tsne(P, Y, n_iter=iters, early_iter=0, exaggeration=exag, learning_rate=lr, n_threads=-1, dof=dof)
    return Y, time.time() - t

ROUND = int(sys.argv[2]) if len(sys.argv) > 2 else 1
sgd_cfgs = [("default", {}), ("mid0.3", {"mid_weight": 0.3}), ("mid1.0", {"mid_weight": 1.0}), ("mid0.3_e100", {"mid_weight": 0.3, "n_epochs": 100})]
if ROUND == 2:
    sgd_cfgs = [("mid1.0_e100", {"mid_weight": 1.0, "n_epochs": 100}), ("mid1.0_e150", {"mid_weight": 1.0, "n_epochs": 150}), ("mid0.5", {"mid_weight": 0.5}), ("mid2.0", {"mid_weight": 2.0}), ("mid1.0_seed1", {"mid_weight": 1.0, "random_state": 1}), ("default_seed1", {"random_state": 1})]
variants = [("base", {}), ("s22", {"scale": 22.0}), ("s40", {"scale": 40.0}), ("lr/2", {"lr": n / 24}), ("lr*2", {"lr": n / 6}),
            ("it450", {"iters": 450}), ("dof0.8", {"dof": 0.8}), ("exag1.2", {"exag": 1.2})]
for sname, skw in sgd_cfgs:
    t = time.time(); m = MAGNET(random_state=0, **skw); Y0 = m.fit_transform(data); t_sgd = time.time() - t
    r = evaluate(data, Y0, labels, cache_key=name)
    log(dict(sgd=sname, variant="sgd_only", t=round(t_sgd, 1), knn=round(r["knn15_preservation"], 3), trust=round(r["trustworthiness"], 3), sp=round(r["dist_spearman"], 3), tri=round(r["triplet_acc"], 3)))
    if ROUND == 1:
        vs = variants if sname in ("default", "mid1.0") else variants[:1] + variants[5:6]
    else:
        vs = variants[:1] + ([variants[2]] if "e1" in sname else [])
    for vname, vkw in vs:
        Y, t_ex = exact(m.graph_, Y0, **vkw)
        r = evaluate(data, Y, labels, cache_key=name)
        log(dict(sgd=sname, variant=vname, t=round(t_sgd + t_ex, 1), knn=round(r["knn15_preservation"], 3), trust=round(r["trustworthiness"], 3), sp=round(r["dist_spearman"], 3), tri=round(r["triplet_acc"], 3)))
