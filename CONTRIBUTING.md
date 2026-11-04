# Contributing to MAGNET

Thanks for considering a contribution. Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/kerighan/lib-magnet/issues).

## Reporting a problem
Open an issue with the MAGNET version (`pip show magnet-learn`), your Python and compiler versions, a minimal
script that reproduces the problem (a synthetic array is fine), and the full traceback or the unexpected output.

## Development setup
```bash
git clone https://github.com/kerighan/lib-magnet && cd lib-magnet
pip install -e ".[test]"            # builds the C++ extension; a C++17 compiler is needed
python setup.py build_ext --inplace --force   # after editing anything in src/ (headers are not tracked by setuptools)
python -m pytest tests -q
```

## Pull requests
- Keep one change per pull request and add a test in `tests/test_magnet.py` for any new behaviour.
- A change of default must come with the measurement that motivates it (the scripts in `benchmarks/` and
  `benchmarks/theory/` produce every number of the README and the papers; quote the script and its output).
- Negative results are welcome too: the companion report keeps what did not work.

## Getting help
Use the issue tracker for questions about usage as well; label them `question`.
