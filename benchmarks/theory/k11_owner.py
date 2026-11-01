"""k11: parallel="owner" against hogwild. Usage: python benchmarks/theory/k11_owner.py sweep|validate (run from the repo root).
sweep: MNIST 70k owner mode (the attraction factor is a constant in src/layout.cpp; the sweep was run through an
environment override that is not kept). validate: hogwild / owner on Fashion-MNIST, 20 newsgroups, blobs, SBM and the
MNIST hybrid. The SGD-only timings and the per-thread profile (MAGNET_PROFILE=1) come from the harness in the log."""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np
from datasets import LOADERS
from metrics import evaluate
from magnet import MAGNET
def full(nm, data, labels, name, graph=False, **kw):
    t = time.time(); Y = MAGNET(random_state=0, **kw).fit_transform(data); dt = time.time() - t
    r = evaluate(None if graph else data, Y, labels, cache_key=name)
    if graph: print(f"{nm:36s} {dt:5.2f}s  acc {r['knn_acc']:.3f}", flush=True)
    else: print(f"{nm:36s} {dt:5.2f}s  knn {r['knn15_preservation']:.3f}  trust {r['trustworthiness']:.3f}  sp {r['dist_spearman']:.3f}  tri {r['triplet_acc']:.3f}" + (f"  acc {r['knn_acc']:.3f}" if 'knn_acc' in r else ""), flush=True)
stage = sys.argv[1]
if stage == "sweep":
    data, labels, name = LOADERS["mnist"](70000)
    full("mnist owner", data, labels, name, parallel="owner")
else:
    for ds in ["fmnist", "news20", "blobs", "sbm"]:
        data, labels, name = LOADERS[ds](70000) if ds == "fmnist" else LOADERS[ds]()
        kw = {"metric": "cosine"} if ds == "news20" else {}
        for par in ("hogwild", "owner"):
            full(f"{ds} {par}", data, labels, name, graph=(ds == "sbm"), parallel=par, **kw)
    data, labels, name = LOADERS["mnist"](70000)
    for par in ("hogwild", "owner"):
        full(f"mnist hybrid {par}", data, labels, name, loss="hybrid", parallel=par)
