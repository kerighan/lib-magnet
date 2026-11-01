"""LaTeX tables of the paper from benchmarks/BENCHMARK.md (booktabs) -> paper/tables/<dataset>.tex."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
md = open(os.path.join(ROOT, "benchmarks", "BENCHMARK.md")).read()
KEYS = {"MNIST": "mnist", "Fashion-MNIST": "fmnist", "20 newsgroups": "news20", "15 Gaussian blobs": "blobs"}
HEAD = ["method", "time (s)", "15-NN pres.", "trust.", "dist. Spearman", "triplet acc.", "10-NN acc.", "compactness*"]
COLS = {"mnist": 7, "fmnist": 7, "news20": 7, "blobs": 5}   # columns kept (blobs has no labels)

def cell(x):
    x = x.strip()
    m = re.match(r"\*\*(.*)\*\*", x)
    return r"\textbf{%s}" % m.group(1) if m else x

def name(x):
    x = x.replace('loss="tsne"', r'\texttt{loss="tsne"}').replace('loss="hybrid"', r'\texttt{loss="hybrid"}')
    return x.replace("_", r"\_").replace("(umap-learn 0.5.12)", "").replace("(GPU, RTX 2070)", "(GPU)")

for sec in re.split(r"^## ", md, flags=re.M)[1:]:
    title, body = sec.split("\n", 1)
    key = next((v for k, v in KEYS.items() if title.startswith(k)), None)
    if key is None:
        continue
    rows = [l for l in body.splitlines() if l.startswith("|")][2:]
    nc = COLS[key]
    out = [r"\begin{tabular}{l" + "r" * (nc - 1) + "}", r"\toprule",
           " & ".join(HEAD[:nc]).replace("compactness*", r"silhouette$^*$") + r" \\", r"\midrule"]
    for l in rows:
        c = [cell(x) for x in l.strip().strip("|").split("|")]
        c[0] = name(c[0])
        out.append(" & ".join(c[:nc]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "tables", key + ".tex"), "w").write("\n".join(out) + "\n")
    print(key, len(rows), "rows")
