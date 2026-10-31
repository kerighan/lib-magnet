"""Law of the kernel tail exponent b: how does the optimum move with the target dimension m,
the intrinsic dimension q of the data and the sample size n?
Stages: A real data x m x b, B synthetic q x m x b, C n x b. Results appended to results_tail_law.jsonl."""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from metrics import evaluate
from magnet import MAGNET

OUT = os.path.join(HERE, "results_tail_law.jsonl")
BS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]


def synthetic(n, q, d=50, clusters=10, seed=0):
    """10 Gaussian clusters, each a q-dimensional Gaussian in a random q-subspace of R^50, no ambient noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, clusters, n)
    centers = rng.normal(0, 8, (clusters, d)).astype(np.float32)
    X = np.empty((n, d), np.float32)
    for c in range(clusters):
        idx = np.where(y == c)[0]
        basis = np.linalg.qr(rng.normal(size=(d, q)))[0].T.astype(np.float32)
        X[idx] = centers[c] + rng.normal(size=(len(idx), q)).astype(np.float32) @ basis
    return X, y


def run(stage, name, X, y, key, m, b, seed, q=None, **kw):
    t = time.time()
    Y = MAGNET(n_components=m, kernel_b=b, repulsion_b=None, random_state=seed, **kw).fit_transform(X)   # coupled exponents, as in the original sweep
    dt = time.time() - t
    r = evaluate(X, Y, y, cache_key=key)
    rec = {"stage": stage, "dataset": name, "n": int(len(X)), "m": m, "q": q, "b": b, "seed": seed, "time": dt, **r}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{stage} {name:12s} n={len(X):6d} m={m} q={q} b={b:<4} s={seed} T={dt:5.1f}s knn15={r['knn15_preservation']:.4f} trust={r['trustworthiness']:.4f} spear={r['dist_spearman']:.4f} trip={r['triplet_acc']:.4f}", flush=True)


def main(stages):
    from datasets import mnist, fmnist, news20
    if "A" in stages and "A2" not in stages:
        for name, (X, y, key), kw in [("mnist", mnist(20000), {}), ("fmnist", fmnist(20000), {}), ("news20", news20(), {"metric": "cosine"})]:
            for m in (1, 2, 3, 4, 5, 6, 8):
                for b in BS:
                    for seed in range(2):
                        run("A", name, X, y, key, m, b, seed, **kw)
    if "A2" in stages:   # news20 beyond 3-D (crashed on component placement the first time)
        X, y, key = news20()
        for m in (4, 5, 6, 8):
            for b in BS:
                for seed in range(2):
                    run("A", "news20", X, y, key, m, b, seed, metric="cosine")
    if "B" in stages:
        for q in (2, 3, 5, 10, 20):
            X, y = synthetic(20000, q)
            for m in (2, 3, 4):
                for b in BS:
                    run("B", f"gauss_q{q}", X, y, f"gaussq{q}_clean", m, b, 0, q=q)
    if "C" in stages:
        for n in (5000, 20000, 70000):
            X, y, key = mnist(n)
            for b in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
                run("C", "mnist", X, y, key, 2, b, 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ABC")
