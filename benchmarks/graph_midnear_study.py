"""Graph-only inputs: does a spectral mid-near term recover global structure? The kNN graph of MNIST
is passed as a sparse adjacency (no data), the layout is evaluated against the hidden data."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from datasets import mnist
from metrics import evaluate
from magnet import MAGNET
from compare import KEYS

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
X, y, name = mnist(n)
P = MAGNET(random_state=0).build_graph(X)
configs = [("walk (graph default)", {}), ("no mid", {"mid_weight": 0.0}), ("spectral pairs w0.1", {"mid_source": "spectral"}),
           ("spectral pairs w0.3", {"mid_source": "spectral", "mid_weight": 0.3}), ("data pairs (reference)", None)]
rows = []
for label, kw in configs:
    vals = []
    for s in range(2):
        t = time.time()
        if kw is None:
            Y = MAGNET(random_state=s).fit_transform(X)
        else:
            Y = MAGNET(input="graph", random_state=s, **kw).fit_transform(P)
        r = evaluate(X, Y, y, cache_key=name); r["time"] = time.time() - t; vals.append(r)
    rows.append((label, vals))
print("config".ljust(26) + "".join(k[:11].rjust(14) for k in KEYS))
for label, vals in rows:
    print(label.ljust(26) + "".join(f"{np.mean([v[k] for v in vals]):14.4f}" for k in KEYS))
