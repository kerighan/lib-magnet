"""openTSNE 1.0.4 on MNIST 70k: where its time goes and what a fair configuration costs.
Phases timed separately: affinities (kNN + perplexity calibration), PCA init, early exaggeration, main
phase. Variants: defaults (perplexity 30 = 90 neighbours, annoy/pynndescent per 'auto', 250 + 500
iterations, lr n/12, FFT); neighbour index; n_jobs 10 vs 20; shorter schedules; uniform k=15 affinities
(MAGNET's graph size). Metrics as in the table. Usage: python benchmarks/opentsne_audit.py"""
import os, sys, time, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from datasets import mnist
from metrics import evaluate
import openTSNE
from openTSNE import affinity, initialization, TSNEEmbedding

X, y, name = mnist(70000)
n = len(X)
# warm-up (numba in pynndescent, openTSNE's own compiled code is C)
affinity.PerplexityBasedNN(np.random.rand(6000, 10).astype(np.float32), perplexity=30, method="pynndescent", n_jobs=-1)

def run(label, aff_kind="perplexity", perplexity=30, k=15, method="auto", n_jobs=-1, ee_iter=250, main_iter=500, lr="auto", exaggeration=12):
    t0 = time.time()
    if aff_kind == "perplexity":
        aff = affinity.PerplexityBasedNN(X, perplexity=perplexity, method=method, n_jobs=n_jobs, random_state=0)
    else:
        aff = affinity.Uniform(X, k_neighbors=k, method=method, n_jobs=n_jobs, random_state=0)
    t1 = time.time()
    init = initialization.pca(X, random_state=0)
    t2 = time.time()
    emb = TSNEEmbedding(init, aff, negative_gradient_method="fft", n_jobs=n_jobs, learning_rate=lr, random_state=0)
    emb = emb.optimize(n_iter=ee_iter, exaggeration=exaggeration, momentum=0.5)
    t3 = time.time()
    emb = emb.optimize(n_iter=main_iter, momentum=0.8)
    t4 = time.time()
    r = evaluate(X, np.asarray(emb, np.float32), y, cache_key=name)
    r.update(t_affinity=round(t1 - t0, 1), t_init=round(t2 - t1, 1), t_ee=round(t3 - t2, 1), t_main=round(t4 - t3, 1), time=round(t4 - t0, 1),
             knn_index=type(aff.knn_index).__name__, config=dict(aff=aff_kind, perplexity=perplexity, k=k, method=method, n_jobs=n_jobs, ee=ee_iter, main=main_iter, lr=str(lr), exaggeration=exaggeration))
    print(f"{label:<34} total {r['time']:6.1f} s = aff {r['t_affinity']:5.1f} ({r['knn_index']}) + init {r['t_init']:4.1f} + EE {r['t_ee']:5.1f} + main {r['t_main']:5.1f} | 15-NN {r['knn15_preservation']:.3f} trust {r['trustworthiness']:.3f} spearman {r['dist_spearman']:.3f} acc {r['knn_acc']:.3f}", flush=True)
    return r

out = {}
out["defaults n_jobs=-1"] = run("defaults, n_jobs=-1 (auto index)")
out["defaults n_jobs=10"] = run("defaults, n_jobs=10", n_jobs=10)
out["pynndescent"] = run("pynndescent index", method="pynndescent")
out["annoy"] = run("annoy index", method="annoy")
out["short 125+250"] = run("125 EE + 250 main", ee_iter=125, main_iter=250)
out["short 250+250"] = run("250 EE + 250 main", main_iter=250)
out["uniform k15"] = run("uniform k=15 affinities, 250+500", aff_kind="uniform", k=15)
out["uniform k15 short"] = run("uniform k=15, 125+250", aff_kind="uniform", k=15, ee_iter=125, main_iter=250)
json.dump(out, open(os.path.join(HERE, "cache", "opentsne_audit.json"), "w"), indent=1)
