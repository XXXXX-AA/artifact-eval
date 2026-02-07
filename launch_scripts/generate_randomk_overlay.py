#!/usr/bin/env python3
"""
generate_randomk_overlay.py

Generate a connected random k-regular overlay bandwidth matrix (N=64 by default)
based on the existing literal bandwidth matrix inside
launch_containernet_mpi_leaf_spine_fixed_v6.py.

Non-edges are set to 0.0 Mbps.

Outputs (saved to current working directory by default):
  - randomk_overlay_matrix.py   (Python literal assignment; drop-in replacement)
  - overlay_randomk64.png       (overlay figure)
  - overlay_randomk64.pdf (optional with --pdf)

Example:
  python3 generate_randomk_overlay.py \
    --src ./launch_containernet_mpi_leaf_spine_fixed_v6.py \
    --N 64 --k 8 --seed 0 \
    --out_matrix randomk_overlay_matrix.py \
    --out_fig overlay_randomk64.png --pdf
"""

import argparse
import ast
import random
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def extract_list_literal_by_brackets(src: str, start_idx: int) -> str:
    depth = 0
    in_str = False
    str_ch = ""
    esc = False
    for i in range(start_idx, len(src)):
        ch = src[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == str_ch:
                in_str = False
            continue
        else:
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return src[start_idx : i + 1]
    raise ValueError("Unmatched brackets while extracting list literal")


def is_square_matrix(x) -> bool:
    if not isinstance(x, list) or not x:
        return False
    if not all(isinstance(r, list) for r in x):
        return False
    n = len(x)
    return all(len(r) == n for r in x)


def find_literal_matrices_in_py(text: str) -> List[Tuple[str, List[List[float]]]]:
    results: List[Tuple[str, List[List[float]]]] = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\b\s*=\s*\[", text):
        name = m.group(1)
        start = m.end() - 1
        try:
            literal = extract_list_literal_by_brackets(text, start)
            val = ast.literal_eval(literal)
            if is_square_matrix(val):
                mat = [[float(v) for v in row] for row in val]
                results.append((name, mat))
        except Exception:
            continue
    return results


def build_connected_random_k_graph(n: int, k: int, seed: int, max_tries: int) -> nx.Graph:
    if n <= 0:
        raise ValueError("N must be positive")
    if k < 0 or k >= n:
        raise ValueError(f"k must be in [0, N-1]. Got k={k}, N={n}")
    if k == 0:
        if n == 1:
            G = nx.Graph()
            G.add_node(0)
            return G
        raise ValueError("k=0 cannot be connected when N>1")
    if k == 1:
        if n == 2:
            G = nx.Graph()
            G.add_nodes_from([0, 1])
            G.add_edge(0, 1)
            return G
        raise ValueError("k=1 cannot be connected when N>2")
    if (n * k) % 2 != 0:
        raise ValueError(f"N*k must be even for k-regular graphs. Got N={n}, k={k}")

    rng = random.Random(seed)

    if k == 2:
        perm = list(range(n))
        rng.shuffle(perm)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            u = perm[i]
            v = perm[(i + 1) % n]
            G.add_edge(u, v)
        return G

    for attempt in range(max_tries):
        attempt_seed = int(seed) + attempt
        G = nx.random_regular_graph(k, n, seed=attempt_seed)
        if nx.is_connected(G):
            return G

    raise RuntimeError(
        f"Failed to generate a connected random {k}-regular graph for N={n} "
        f"after {max_tries} attempts. Try a different seed or larger k."
    )


def build_randomk_overlay_matrix(base: List[List[float]], G: nx.Graph) -> List[List[float]]:
    n = len(base)
    B = [[0.0 for _ in range(n)] for _ in range(n)]
    for u, v in G.edges():
        B[u][v] = float(base[u][v])
        B[v][u] = float(base[v][u])
    return B


def save_matrix_py(out_path: str, var_name: str, B: List[List[float]]):
    lines = [f"{var_name} = ["]
    for row in B:
        row_s = ", ".join(f"{v:.1f}" for v in row)
        lines.append(f"    [{row_s}],")
    lines.append("]")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_randomk_overlay(out_fig: str, G: nx.Graph, k: int, show_labels: bool, pdf: bool):
    n = G.number_of_nodes()
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.axis("off")

    nx.draw_networkx_edges(G, pos, ax=ax, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200)

    if show_labels:
        labels = {i: f"w{i}" for i in range(n)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title(f"Random-k Overlay (k={k}, N={n})")

    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    if pdf:
        pdf_path = str(Path(out_fig).with_suffix(".pdf"))
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to launch_containernet_mpi_leaf_spine_fixed_v6.py")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--k", type=int, required=True, help="Degree k for the random k-regular overlay")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    ap.add_argument("--max_tries", type=int, default=200, help="Max attempts to get a connected graph")
    ap.add_argument("--out_matrix", default="randomk_overlay_matrix.py")
    ap.add_argument("--var_name", default=None, help="Variable name to write (default: base matrix variable name)")
    ap.add_argument("--out_fig", default=None, help="Output figure path (default: overlay_randomk{N}.png)")
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--show_labels", action="store_true")
    args = ap.parse_args()

    text = Path(args.src).read_text(encoding="utf-8", errors="ignore")
    mats = find_literal_matrices_in_py(text)

    base_name, base = None, None
    for name, mat in mats:
        if len(mat) == args.N:
            base_name, base = name, mat
            break

    if base is None:
        dims = sorted({len(mat) for _, mat in mats})
        raise SystemExit(
            f"[error] Could not find a literal {args.N}x{args.N} matrix in {args.src}. "
            f"Found literal square matrices with dims: {dims}."
        )

    G = build_connected_random_k_graph(args.N, args.k, args.seed, args.max_tries)
    B_randomk = build_randomk_overlay_matrix(base, G)

    var_name = args.var_name if args.var_name else base_name
    save_matrix_py(args.out_matrix, var_name, B_randomk)

    out_fig = args.out_fig if args.out_fig else f"overlay_randomk{args.N}.png"
    draw_randomk_overlay(out_fig, G, args.k, args.show_labels, args.pdf)

    print(f"[ok] Extracted base matrix '{base_name}' ({args.N}x{args.N}) from: {args.src}")
    print(f"[ok] Generated connected random-k graph: N={args.N}, k={args.k}, edges={G.number_of_edges()}, seed={args.seed}")
    print(f"[ok] Wrote random-k overlay matrix to: {args.out_matrix} (var name: {var_name})")
    print(f"[ok] Wrote overlay figure to: {out_fig}" + (f" and {Path(out_fig).with_suffix('.pdf')}" if args.pdf else ""))


if __name__ == "__main__":
    main()
