"""Validation of the final defaults on cases they were not tuned on."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from datasets import mnist, sbm_graph
from metrics import evaluate
from magnet import MAGNET
from compare import KEYS


def row(label, vals):
    print(label.ljust(34) + "".join(f"{np.mean([v[k] for v in vals if k in v]):14.4f}" if all(k in v for v in vals) else " " * 14 for k in KEYS), flush=True)


X, y, key = mnist(20000)
print("== 3-D: kernel exponent sweep (MNIST 20k, 2 seeds, n_components=3)")
print("config".ljust(34) + "".join(k[:11].rjust(14) for k in KEYS))
for b in (0.5, 0.6, 0.8, 1.0):
    vals = []
    for s in range(2):
        t = time.time(); Y = MAGNET(n_components=3, kernel_b=b, random_state=s).fit_transform(X)
        r = evaluate(X, Y, y, cache_key=key); r["time"] = time.time() - t; vals.append(r)
    row(f"3D b={b}", vals)

print("== n_neighbors under the new kernel (MNIST 20k, 2 seeds)")
for k in (10, 15, 30):
    vals = []
    for s in range(2):
        t = time.time(); Y = MAGNET(n_neighbors=k, random_state=s).fit_transform(X)
        r = evaluate(X, Y, y, cache_key=key); r["time"] = time.time() - t; vals.append(r)
    row(f"n_neighbors={k}", vals)

print("== compact preset at b=0.6 (MNIST 20k, 2 seeds)")
for label, kw in [("default", {}), ("late_exaggeration=1.5", {"late_exaggeration": 1.5}), ("late_exaggeration=2", {"late_exaggeration": 2.0})]:
    vals = []
    for s in range(2):
        t = time.time(); Y = MAGNET(random_state=s, **kw).fit_transform(X)
        r = evaluate(X, Y, y, cache_key=key); r["time"] = time.time() - t; vals.append(r)
    row(label, vals)

print("== graph input (SBM 30k): label accuracy of the layout")
G, yg, _ = sbm_graph()
for label, kw in [("new defaults", {}), ("old kernel (b None, min_dist 0.1)", {"kernel_b": None, "min_dist": 0.1})]:
    t = time.time(); Y = MAGNET(random_state=0, **kw).fit_transform(G)
    r = evaluate(None, Y, yg); r["time"] = time.time() - t
    print(f"{label:34s} T={r['time']:.1f}s 10-NN acc={r['knn_acc']:.4f} silhouette={r['silhouette']:.3f}")

print("== transform() with the new defaults (MNIST 60k -> 10k)")
Xa, ya, _ = mnist(70000)
from sklearn.neighbors import KNeighborsClassifier
m = MAGNET(random_state=0).fit(Xa[:60000])
t = time.time(); Yt = m.transform(Xa[60000:]); dt = time.time() - t
clf = KNeighborsClassifier(10).fit(m.embedding_, ya[:60000])
print(f"transform 10k: T={dt:.2f}s 10-NN accuracy {(clf.predict(Yt) == ya[60000:]).mean():.4f}")
