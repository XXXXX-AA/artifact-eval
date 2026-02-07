#!/usr/bin/env python3
"""draw_underlay_topology.py

Draw the *underlay* leaf-spine topology used in launch_containernet_mpi_leaf_spine_fixed_v6.py.

Topology (as implemented in build_leaf_spine):
  - N workers (containers) named worker0..worker{N-1}
  - L = ceil(N / tor_fanout) leaf switches: l1..lL
  - S = min(4, ceil(L/2)) spine switches: p1..pS  (clamped to [1,4])
  - Each worker i connects to leaf l_{i // tor_fanout + 1}
  - Each leaf li connects to spine p_{li % S + 1}

This script produces a PNG/SVG/PDF figure for paper/report slides.

Dependencies:
  - networkx
  - matplotlib

Examples:
  python3 draw_underlay_topology.py --N 64 --tor_fanout 4 --out underlay64.png
  python3 draw_underlay_topology.py --N 64 --tor_fanout 4 --out underlay64.pdf --show-worker-labels

Notes:
  - By default we label only switches (spines/leaves) to keep the figure readable.
"""

import argparse
import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx

def build_graph(N: int, tor_fanout: int, S: int | None = None):
    if N <= 0:
        raise ValueError("N must be positive")
    if tor_fanout <= 0:
        raise ValueError("tor_fanout must be positive")

    L = int(math.ceil(N / float(tor_fanout)))
    if S is None:
        S = max(1, int(min(4, math.ceil(L / 2.0))))
    else:
        S = max(1, int(S))

    G = nx.Graph()

    spines = [f"p{i+1}" for i in range(S)]
    leaves = [f"l{i+1}" for i in range(L)]
    workers = [f"w{i}" for i in range(N)]

    for n in spines + leaves + workers:
        G.add_node(n)

    # worker -> leaf
    for i in range(N):
        leaf = leaves[i // tor_fanout]
        G.add_edge(workers[i], leaf)

    # leaf -> spine (one spine per leaf, round-robin)
    for li in range(L):
        spine = spines[li % S]
        G.add_edge(leaves[li], spine)

    meta = {
        "N": N,
        "tor_fanout": tor_fanout,
        "L": L,
        "S": S,
        "spines": spines,
        "leaves": leaves,
        "workers": workers,
    }
    return G, meta

def layout_positions(meta) -> Dict[str, Tuple[float, float]]:
    """Manual layered layout: spines on top, leaves middle, workers bottom."""
    N, L, S = meta["N"], meta["L"], meta["S"]
    spines, leaves, workers = meta["spines"], meta["leaves"], meta["workers"]

    pos: Dict[str, Tuple[float, float]] = {}

    # x coordinates
    # Spread spines across [0, 1]
    for k, s in enumerate(spines):
        x = (k + 1) / (S + 1)
        pos[s] = (x, 2.0)

    # Spread leaves across [0, 1]
    for k, l in enumerate(leaves):
        x = (k + 1) / (L + 1)
        pos[l] = (x, 1.0)

    # Workers: grouped under their leaf
    # Place tor_fanout workers under each leaf with small horizontal jitter
    jitter = 0.008  # small; avoids overlap when labels off
    for i, w in enumerate(workers):
        leaf_idx = i // meta["tor_fanout"]
        lx, _ = pos[leaves[leaf_idx]]
        # position within the group
        within = i % meta["tor_fanout"]
        dx = (within - (meta["tor_fanout"] - 1) / 2.0) * jitter
        pos[w] = (lx + dx, 0.0)

    return pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64, help="Number of worker nodes (default: 64)")
    ap.add_argument("--tor_fanout", type=int, default=4, help="Workers per leaf (default: 4)")
    ap.add_argument("--S", type=int, default=None, help="Number of spines. Default follows v6 rule min(4, ceil(L/2)).")
    ap.add_argument("--out", type=str, default="underlay64.png", help="Output figure path (.png/.pdf/.svg)")
    ap.add_argument("--dpi", type=int, default=200, help="DPI for raster outputs (default: 200)")
    ap.add_argument("--show-worker-labels", action="store_true", help="Show worker labels (very dense for N=64)")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title")
    args = ap.parse_args()

    G, meta = build_graph(args.N, args.tor_fanout, args.S)
    meta["tor_fanout"] = args.tor_fanout
    pos = layout_positions(meta)

    # Draw
    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.axis('off')

    spines = meta["spines"]
    leaves = meta["leaves"]
    workers = meta["workers"]

    # Edges first
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.8)

    # Nodes: draw in layers with different sizes (no explicit colors)
    nx.draw_networkx_nodes(G, pos, nodelist=workers, ax=ax, node_size=40)
    nx.draw_networkx_nodes(G, pos, nodelist=leaves, ax=ax, node_size=300)
    nx.draw_networkx_nodes(G, pos, nodelist=spines, ax=ax, node_size=380)

    # Labels: switches only by default
    labels = {n: n for n in spines + leaves}
    if args.show_worker_labels:
        labels.update({w: w for w in workers})
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    title = args.title
    if title is None:
        title = f"Underlay leaf-spine topology: N={meta['N']}, L={meta['L']}, S={meta['S']}, fanout={meta['tor_fanout']}"
    ax.set_title(title)

    out = args.out
    if out.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        plt.savefig(out, dpi=args.dpi, bbox_inches='tight')
    else:
        plt.savefig(out, bbox_inches='tight')
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
