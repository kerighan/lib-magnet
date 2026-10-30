import os
from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

compile_args = ["-O3", "-pthread"]
if os.environ.get("MAGNET_NATIVE"):  # opt-in: faster, but the binary is tied to this CPU
    compile_args.append("-march=native")

ext_modules = [
    Pybind11Extension(
        "_magnet",
        sorted(glob("src/*.cpp")),
        extra_compile_args=compile_args,
        extra_link_args=["-pthread"],
        cxx_std=17,
    ),
]

setup(
    name="magnet-learn",
    version="3.0.0",
    description="MAGNET: fast graph-based dimensionality reduction (kNN graph + random walks + SGD)",
    packages=find_packages(exclude=("benchmarks", "tests")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=["numpy", "scipy", "tqdm"],
    extras_require={"hnsw": ["hnswlib"]},
    python_requires=">=3.8",
)
