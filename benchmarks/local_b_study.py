"""Per-point kernel exponent from the local intrinsic dimension: does local adaptation beat the best
constant b? Candidate laws b_i = b0 (median_m / m_i)^p, scaled so that the median exponent is b0."""
import sys, os, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
sys.path.insert(0, "/home/maixent/Dropbox (Reputation Squad)/DATAPROJECTS/r&d/dim_fingerprinting/dimprint/src")
import numpy as np
from datasets import mnist, fmnist
from metrics import evaluate
from magnet import MAGNET
from magnet.roles import local_intrinsic_dimension
from compare import KEYS

for name, (X, y, key) in [("mnist20k", mnist(20000)), ("fmnist20k", fmnist(20000))]:
    m = local_intrinsic_dimension(X, 15)
    med = float(np.median(m))
    print(f"== {name}: local ID median {med:.1f}, p10 {np.percentile(m,10):.1f}, p90 {np.percentile(m,90):.1f}")
    configs = [("const b0.5", 0.5), ("const b0.6", 0.6)]
    for b0 in (0.5, 0.6):
        for p in (0.5, 1.0):
            configs.append((f"local b0={b0} p={p}", np.clip(b0 * (med / m) ** p, 0.3, 1.0).astype(np.float32)))
        configs.append((f"local b0={b0} p=-0.5 (inverse)", np.clip(b0 * (med / m) ** -0.5, 0.3, 1.0).astype(np.float32)))
    print("config".ljust(30) + "".join(k[:11].rjust(14) for k in KEYS))
    for label, kb in configs:
        vals = []
        for s in range(2):
            Y = MAGNET(kernel_b=kb, random_state=s).fit_transform(X)
            r = evaluate(X, Y, y, cache_key=key); r["time"] = 0.0; vals.append(r)
        print(label.ljust(30) + "".join(f"{np.mean([v[k] for v in vals]):14.4f}" for k in KEYS), flush=True)
