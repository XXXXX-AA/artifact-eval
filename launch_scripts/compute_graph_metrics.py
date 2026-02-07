#!/usr/bin/env python3
import argparse
import ast
from collections import deque


def load_matrices(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    tree = ast.parse(data, filename=path)
    mats = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        name = target.id
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(value, list) and value and isinstance(value[0], list):
            mats.setdefault(name, []).append(value)
    return mats


def build_adj(matrix):
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("Matrix must be square")
    adj = [[] for _ in range(n)]
    radj = [[] for _ in range(n)]
    edges = 0
    for i in range(n):
        for j, v in enumerate(matrix[i]):
            if i == j:
                continue
            if float(v) > 0.0:
                adj[i].append(j)
                radj[j].append(i)
                edges += 1
    return adj, radj, edges


def bfs(start, adj):
    n = len(adj)
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def is_strongly_connected(adj, radj):
    n = len(adj)
    if n == 0:
        return True
    d1 = bfs(0, adj)
    if any(d < 0 for d in d1):
        return False
    d2 = bfs(0, radj)
    if any(d < 0 for d in d2):
        return False
    return True


def shortest_path_stats(adj):
    n = len(adj)
    total_pairs = n * (n - 1)
    reachable_pairs = 0
    dist_sum = 0
    max_dist = 0
    for i in range(n):
        dist = bfs(i, adj)
        for j in range(n):
            if i == j:
                continue
            d = dist[j]
            if d >= 0:
                reachable_pairs += 1
                dist_sum += d
                if d > max_dist:
                    max_dist = d
    avg = (dist_sum / reachable_pairs) if reachable_pairs > 0 else float("inf")
    return {
        "total_pairs": total_pairs,
        "reachable_pairs": reachable_pairs,
        "avg": avg,
        "max_dist": max_dist if reachable_pairs > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute graph metrics from bandwidth matrix.")
    parser.add_argument("--src", required=True, help="Path to matrix .py file")
    parser.add_argument("--var", default=None, help="Variable name of matrix list")
    parser.add_argument("--index", type=int, default=0, help="Which matrix to use if multiple")
    args = parser.parse_args()

    mats = load_matrices(args.src)
    if not mats:
        raise RuntimeError("No matrix lists found in file")

    if args.var is None:
        if len(mats) == 1:
            var_name = next(iter(mats.keys()))
        else:
            names = ", ".join(sorted(mats.keys()))
            raise RuntimeError(f"Multiple matrix vars found. Use --var. Available: {names}")
    else:
        var_name = args.var
        if var_name not in mats:
            names = ", ".join(sorted(mats.keys()))
            raise RuntimeError(f"Var '{var_name}' not found. Available: {names}")

    matrices = mats[var_name]
    if args.index < 0 or args.index >= len(matrices):
        raise RuntimeError(f"index out of range: {args.index} (found {len(matrices)} matrices for {var_name})")

    matrix = matrices[args.index]
    n = len(matrix)
    adj, radj, edges = build_adj(matrix)

    sc = is_strongly_connected(adj, radj)
    stats = shortest_path_stats(adj)

    print(f"Matrix: {args.src}")
    print(f"Var: {var_name} (index {args.index})")
    print(f"Nodes: {n}")
    print(f"Directed edges (bw>0, off-diagonal): {edges}")
    print(f"Strongly connected: {sc}")

    if sc:
        print(f"Diameter (hop count): {stats['max_dist']}")
        print(f"Avg shortest path (hop count): {stats['avg']:.6f}")
    else:
        print("Diameter (hop count): inf (not strongly connected)")
        if stats["reachable_pairs"] > 0:
            ratio = stats["reachable_pairs"] / stats["total_pairs"]
            print(f"Avg shortest path over reachable pairs: {stats['avg']:.6f}")
            print(f"Reachable pairs: {stats['reachable_pairs']} / {stats['total_pairs']} ({ratio:.3f})")
        else:
            print("Avg shortest path over reachable pairs: inf (no reachable pairs)")


if __name__ == "__main__":
    main()
