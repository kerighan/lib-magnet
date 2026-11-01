"""k9: mid_weight of the SGD pass under loss="hybrid" on Fashion-MNIST 70k, 20 newsgroups and blobs (0.1 / 0.5 / 1.0).
Usage: python benchmarks/theory/k9_mid_validate.py OUT.jsonl ; results in k9_mid_validate.log."""
import sys, os, time, json
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE); sys.path.insert(0, os.path.dirname(B)); sys.path.insert(0, B)
import numpy as np
from datasets import LOADERS
from metrics import evaluate
from magnet import MAGNET
out = open(sys.argv[1], "a")
for ds in ["fmnist", "news20", "blobs"]:
    data, labels, name = LOADERS[ds](70000) if ds == "fmnist" else LOADERS[ds]()
    for w in [0.1, 0.5, 1.0]:
        kw = {"metric": "cosine"} if ds == "news20" else {}
        t = time.time(); Y = MAGNET(loss="hybrid", mid_weight=w, random_state=0, **kw).fit_transform(data); dt = time.time() - t
        r = evaluate(data, Y, labels, cache_key=name)
        row = dict(ds=ds, mid=w, t=round(dt, 1), knn=round(r["knn15_preservation"], 3), trust=round(r["trustworthiness"], 3), sp=round(r["dist_spearman"], 3), tri=round(r["triplet_acc"], 3))
        print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()
