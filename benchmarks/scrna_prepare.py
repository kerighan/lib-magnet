"""Single-cell RNA-seq data sets for the held-out evaluation (k47), prepared the standard way (Kobak & Berens 2019):
library-size normalisation to counts per million, log1p, the 3000 genes of highest variance of the log values, PCA
to 50 components; cells without a cell-type annotation are dropped. Streams the dense Tasic CSVs in chunks so that
memory stays below ~1.5 GB.
  pbmc68k  : Zheng et al. 2017, 10x fresh 68k PBMCs (donor A), the 11 annotated cell types of the paper's repository
  tasic2018: Tasic et al. 2018, mouse VISp + ALM (Smart-seq2), exon counts; labels = the annotated subclasses
             (sample_heatmap_plot_data.csv of berenslab/rna-seq-tsne), cells with class "Low Quality" / "No Class" dropped
-> benchmarks/cache/scrna/<name>.npz with X (n, 50) float32, y (n,) int, names. Data are read from
benchmarks/cache/scrna/<name>/ (downloaded files only; no code lives there). Usage: python -I benchmarks/scrna_prepare.py [pbmc68k] [tasic2018]"""
import os, sys, io, tarfile, zipfile
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "cache", "scrna")
import numpy as np, scipy.sparse as sp, scipy.io
from sklearn.decomposition import PCA


def finish(name, counts, y, names, n_genes=3000, n_pcs=50):
    """counts: (cells, genes) sparse CSR of raw counts."""
    counts = sp.csr_matrix(counts, dtype=np.float32)
    lib = np.asarray(counts.sum(1)).ravel(); keep = lib > 0
    counts, y, lib = counts[keep], y[keep], lib[keep]
    X = sp.diags(1e6 / lib) @ counts; X.data = np.log1p(X.data)
    mean = np.asarray(X.mean(0)).ravel(); sq = np.asarray(X.multiply(X).mean(0)).ravel(); var = sq - mean ** 2
    genes = np.argsort(-var)[:n_genes]
    Z = X[:, genes].toarray(); Z -= Z.mean(0)
    P = PCA(n_pcs, svd_solver="randomized", random_state=0).fit_transform(Z).astype(np.float32)
    out = os.path.join(D, f"{name}.npz"); np.savez(out, X=P, y=y.astype(np.int32), names=np.array(names))
    print(f"{name}: {P.shape[0]} cells, {len(names)} classes, PCA-{n_pcs} on the top {n_genes} genes -> {out}", flush=True)


def pbmc68k():
    base = os.path.join(D, "pbmc68k")
    with tarfile.open(os.path.join(base, "matrices.tar.gz")) as t:
        members = {os.path.basename(m.name): m for m in t.getmembers() if m.isfile()}
        M = scipy.io.mmread(io.BytesIO(t.extractfile(members["matrix.mtx"]).read())).T.tocsr()   # cells x genes
        barcodes = [l.strip() for l in t.extractfile(members["barcodes.tsv"]).read().decode().splitlines()]
    import pandas as pd
    a = pd.read_csv(os.path.join(base, "annotation.tsv"), sep="\t"); ann = dict(zip(a["barcodes"], a["celltype"]))
    names = sorted(set(ann.values())); idx = {c: k for k, c in enumerate(names)}
    rows = [k for k, b in enumerate(barcodes) if b in ann]
    y = np.array([idx[ann[barcodes[k]]] for k in rows])
    print(f"pbmc68k: {M.shape[0]} cells x {M.shape[1]} genes, {len(rows)} annotated", flush=True)
    finish("pbmc68k", M[rows], y, names)


def tasic2018():
    import pandas as pd
    base = os.path.join(D, "tasic2018")
    ann = pd.read_csv(os.path.join(base, "sample_heatmap_plot_data.csv"))
    print("tasic annotation columns:", list(ann.columns)[:12], flush=True)
    ann = ann[~ann["class_label"].isin(["Low Quality", "No Class"])]
    AREAS = {"ALM", "VISp"}
    def subclass(c):
        """Subclass from the cluster label: the first token, or the first two for layer-named excitatory types
        ("L5 PT ALM Slco2a1" -> "L5 PT", "L2/3 IT VISp Rrad" -> "L2/3 IT", "L6b VISp Col8a1" -> "L6b", "Sst Chodl" -> "Sst")."""
        t = c.split()
        if t[0][0] == "L" and t[0][1:2].isdigit() and len(t) > 1 and t[1] not in AREAS and t[1].isupper(): return " ".join(t[:2])
        return t[0]
    label = dict(zip(ann["sample_name"], [subclass(c) for c in ann["cluster_label"]]))
    print("tasic subclasses:", sorted(set(label.values())), flush=True)
    def chunks(z, cols=None):
        """Stream the exon matrix of one archive: yields (gene names, genes x kept-cells float32 block)."""
        with zipfile.ZipFile(os.path.join(base, z)) as zf:
            exon = [n for n in zf.namelist() if "exon-matrix" in n][0]
            with io.TextIOWrapper(zf.open(exon)) as f:
                f.readline()
                for chunk in pd.read_csv(f, header=None, chunksize=2000, index_col=0):
                    yield chunk.index.astype(str).to_numpy(), chunk.to_numpy(dtype=np.float32)[:, cols]
    files = ("visp.zip", "alm.zip"); cols, ys = {}, []
    for z in files:
        with zipfile.ZipFile(os.path.join(base, z)) as zf:
            exon = [n for n in zf.namelist() if "exon-matrix" in n][0]
            with io.TextIOWrapper(zf.open(exon)) as f: cells = [c.strip('"') for c in f.readline().rstrip("\n").split(",")[1:]]
        cols[z] = np.array([k for k, c in enumerate(cells) if c in label]); ys += [label[cells[k]] for k in cols[z]]
    # pass 1: library sizes of the kept cells
    lib = {z: sum(b.sum(0) for _, b in chunks(z, cols[z])) for z in files}
    # pass 2: per-gene variance of log1p(CPM) over all kept cells
    s1, s2, gene_names, n = None, None, None, sum(len(cols[z]) for z in files)
    for z in files:
        a1, a2, g = [], [], []
        for names_, b in chunks(z, cols[z]):
            v = np.log1p(b * (1e6 / lib[z])); a1.append(v.sum(1)); a2.append((v * v).sum(1)); g.append(names_)
        a1, a2, g = np.concatenate(a1), np.concatenate(a2), np.concatenate(g)
        if gene_names is None: gene_names, s1, s2 = g, a1, a2
        else: assert (g == gene_names).all(), "gene order differs between VISp and ALM"; s1, s2 = s1 + a1, s2 + a2
    var = s2 / n - (s1 / n) ** 2; top = np.sort(np.argsort(-var)[:3000])
    # pass 3: extract the selected genes, normalised and logged, then PCA
    Z = []
    for z in files:
        rows, off = [], 0
        for _, b in chunks(z, cols[z]):
            sel = top[(top >= off) & (top < off + len(b))] - off; off += len(b)
            if len(sel): rows.append(np.log1p(b[sel] * (1e6 / lib[z])))
        Z.append(np.concatenate(rows).T)
        print(f"tasic {z}: {len(cols[z])} annotated cells", flush=True)
    Z = np.concatenate(Z).astype(np.float32); Z -= Z.mean(0)
    names = sorted(set(ys)); idx = {c: k for k, c in enumerate(names)}; y = np.array([idx[c] for c in ys])
    P = PCA(50, svd_solver="randomized", random_state=0).fit_transform(Z).astype(np.float32)
    out = os.path.join(D, "tasic2018.npz"); np.savez(out, X=P, y=y.astype(np.int32), names=np.array(names))
    print(f"tasic2018: {P.shape[0]} cells, {len(names)} classes, PCA-50 on the top 3000 genes -> {out}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["pbmc68k", "tasic2018"]
    for w in which: {"pbmc68k": pbmc68k, "tasic2018": tasic2018}[w]()
