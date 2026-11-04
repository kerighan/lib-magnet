"""Best / second marks for the comparison tables: the values are compared at the precision they are printed with,
so ties at that precision share the mark; best in bold, second in underline (LaTeX) or <ins> (markdown)."""
import math


def marks(values, higher=True, decimals=3):
    """values: list of floats (nan / None allowed). Returns a list of 'best', 'second' or None."""
    r = [None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), decimals) for v in values]
    distinct = sorted({v for v in r if v is not None}, reverse=higher)
    best = distinct[0] if distinct else None
    second = distinct[1] if len(distinct) > 1 else None
    return [None if v is None else ("best" if v == best else ("second" if v == second else None)) for v in r]


def latex(txt, mark):
    return r"\textbf{%s}" % txt if mark == "best" else (r"\underline{%s}" % txt if mark == "second" else txt)


def markdown(txt, mark):
    return f"**{txt}**" if mark == "best" else (f"<ins>{txt}</ins>" if mark == "second" else txt)
