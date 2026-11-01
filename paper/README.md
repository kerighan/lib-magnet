# Paper outline (arXiv)

`main.tex` is the outline: section bodies are bullet lists of claims with the numbers that back them,
`\todo{}` marks what is missing. Figures and tables are generated from the repository:

```bash
python paper/make_fig_layouts.py mnist     # figures/layouts_mnist.png (runs UMAP, openTSNE, PaCMAP, MAGNET x3; ~3 min)
python paper/make_fig_layouts.py fmnist
python paper/make_fig_ablations.py          # figures/exponents.png, hybrid.png, scaling.png (from recorded runs)
python paper/make_tables.py                 # tables/*.tex from benchmarks/BENCHMARK.md
latexmk -pdf main.tex        # or: tectonic -X compile main.tex (conda env "tex" on this machine)
```

`figures/tradeoff.png` and `figures/graph_demo.png` are copies of `benchmarks/plots/`.
