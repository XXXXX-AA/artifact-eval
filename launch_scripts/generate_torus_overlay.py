#!/usr/bin/env python3
"""
generate_torus_overlay.py

Generate a 2D Torus overlay bandwidth matrix (N=64 by default) based on the
existing literal bandwidth matrix inside launch_containernet_mpi_leaf_spine_fixed_v6.py,
and draw the Torus overlay graph.

Outputs (saved to current working directory by default):
  - torus_overlay_matrix.py      (Python literal assignment; drop-in replacement)
  - overlay_torus64.png          (overlay figure)
  - overlay_torus64.pdf (optional with --pdf)

Example:
  python3 generate_torus_overlay.py \
    --src ./launch_containernet_mpi_leaf_spine_fixed_v6.py \
    --N 64 --rows 8 --cols 8 \
    --out_matrix torus_overlay_matrix.py \
    --out_fig overlay_torus64.png --pdf
"""

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple, Optional

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


def torus_neighbors(idx: int, rows: int, cols: int) -> List[int]:
    r = idx // cols
    c = idx % cols
    up = ((r - 1) % rows) * cols + c
    down = ((r + 1) % rows) * cols + c
    left = r * cols + ((c - 1) % cols)
    right = r * cols + ((c + 1) % cols)
    return [up, down, left, right]


def compute_default_baseline(base: List[List[float]]) -> float:
    vals = []
    n = len(base)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = float(base[i][j])
            if v > 0:
                vals.append(v)
    if not vals:
        return 0.1
    m = min(vals)
    # keep baseline small but non-zero; also avoid super tiny 1kbit effects
    return max(0.1, min(1.0, m))


def build_torus_overlay_matrix(base: List[List[float]], rows: int, cols: int, baseline_mbps: float) -> List[List[float]]:
    N = len(base)
    if rows * cols != N:
        raise ValueError(f"rows*cols must equal N. Got {rows}*{cols}={rows*cols}, N={N}")

    B = [[baseline_mbps for _ in range(N)] for _ in range(N)]
    for i in range(N):
        B[i][i] = 0.0
        for j in torus_neighbors(i, rows, cols):
            B[i][j] = float(base[i][j])  # keep original directional bandwidth for neighbors
    return B


def save_matrix_py(out_path: str, var_name: str, B: List[List[float]]):
    lines = [f"{var_name} = ["]
    for row in B:
        row_s = ", ".join(f"{v:.1f}" for v in row)
        lines.append(f"    [{row_s}],")
    lines.append("]")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_torus_overlay(out_fig: str, rows: int, cols: int, show_labels: bool, pdf: bool):
    N = rows * cols
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in torus_neighbors(i, rows, cols):
            if i < j:
                G.add_edge(i, j)

    pos = {i: (i % cols, -(i // cols)) for i in range(N)}

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.axis("off")

    nx.draw_networkx_edges(G, pos, ax=ax, width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=200)

    if show_labels:
        labels = {i: f"w{i}" for i in range(N)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title(f"2D Torus Overlay (rows={rows}, cols={cols}, N={N})")

    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    if pdf:
        pdf_path = str(Path(out_fig).with_suffix(".pdf"))
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to launch_containernet_mpi_leaf_spine_fixed_v6.py")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--baseline", type=float, default=None, help="Baseline Mbps for non-neighbors (default: min-positive clipped to [0.1,1.0])")
    ap.add_argument("--out_matrix", default="torus_overlay_matrix.py")
    ap.add_argument("--var_name", default=None, help="Variable name to write (default: base matrix variable name)")
    ap.add_argument("--out_fig", default="overlay_torus64.png")
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

    baseline = args.baseline if args.baseline is not None else compute_default_baseline(base)
    B_torus = build_torus_overlay_matrix(base, args.rows, args.cols, baseline)

    var_name = args.var_name if args.var_name else base_name
    save_matrix_py(args.out_matrix, var_name, B_torus)
    draw_torus_overlay(args.out_fig, args.rows, args.cols, args.show_labels, args.pdf)

    print(f"[ok] Extracted base matrix '{base_name}' ({args.N}x{args.N}) from: {args.src}")
    print(f"[ok] Wrote Torus overlay matrix to: {args.out_matrix} (var name: {var_name})")
    print(f"[ok] Baseline (non-neighbors): {baseline:.3f} Mbps")
    print(f"[ok] Wrote overlay figure to: {args.out_fig}" + (f" and {Path(args.out_fig).with_suffix('.pdf')}" if args.pdf else ""))


if __name__ == "__main__":
    main()
