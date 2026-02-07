#!/usr/bin/env python3
"""sanity_check_extract.py

Sanity-check a Containernet/Mininet bandwidth matrix enforced by Linux tc (HTB + netem),
and **extract the matrix directly from an arbitrary Python source file** even if the
assignment appears inside a function (indented), e.g.:

    def foo():
        bandwidth_list15 = [[...], ...]

This script does NOT execute the source file; it parses the literal list via:
  1) AST scan for module-level assignments (fast path)
  2) Regex search + bracket matching for 'var_name = [' anywhere in the file (fallback)

Example:
  python3 sanity_check_extract.py \
    --matrix-py ./launch_containernet_mpi_leaf_spine_fixed_v6.py \
    --matrix-var bandwidth_list15 \
    --n 15 --pairs 10 --duration 5 --seed 0
"""

import argparse
import ast
import json
import random
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

def run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

def docker_exec(container: str, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return run(["docker", "exec", container] + args, timeout=timeout)

def docker_exec_py(container: str, code: str, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return docker_exec(container, ["python3", "-c", code] + args, timeout=timeout)

def _extract_list_literal_by_brackets(src: str, start_idx: int) -> str:
    """Return the complete bracketed list literal starting at src[start_idx] == '['."""
    depth = 0
    in_str = False
    str_ch = ''
    esc = False
    for i in range(start_idx, len(src)):
        ch = src[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == str_ch:
                in_str = False
            continue
        else:
            if ch in ('"', "'"):
                in_str = True
                str_ch = ch
                continue
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return src[start_idx:i+1]
    raise ValueError("Unmatched brackets while extracting list literal")

def load_matrix_from_py(path: str, var_name: str) -> List[List[float]]:
    """Load a 2D list literal assigned to var_name from a Python file without executing it."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    # (1) AST: module-level assignment
    try:
        tree = ast.parse(text, filename=path)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == var_name:
                        val = ast.literal_eval(node.value)
                        if not isinstance(val, list) or not val or not isinstance(val[0], list):
                            raise ValueError(f"{var_name} is not a 2D list")
                        return [[float(x) for x in row] for row in val]
    except Exception:
        pass

    # (2) Regex + bracket matching: find var_name anywhere (indented OK)
    m = re.search(rf"\b{re.escape(var_name)}\b\s*=\s*\[", text)
    if not m:
        raise ValueError(
            f"Could not find '{var_name} = [[...]]' in {path}.\n"
            f"Tip: ensure the matrix is written as a literal list, not computed at runtime."
        )
    start = m.end() - 1  # index of '['
    literal = _extract_list_literal_by_brackets(text, start)
    val = ast.literal_eval(literal)
    if not isinstance(val, list) or not val or not isinstance(val[0], list):
        raise ValueError(f"{var_name} is not a 2D list")
    return [[float(x) for x in row] for row in val]

def overlay_ip(ip_prefix: str, idx: int) -> str:
    return f"{ip_prefix}{idx+1}"

RECEIVER_CODE = """import socket, time, json, sys
dur=float(sys.argv[1]); port=int(sys.argv[2])
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", port)); s.listen(1)
c,_=s.accept(); c.settimeout(1.0)
t0=time.time(); n=0
while time.time()-t0<dur:
    try: d=c.recv(65536)
    except socket.timeout: continue
    if not d: break
    n+=len(d)
print(json.dumps({"mbps": n*8/dur/1e6, "bytes": n}))
"""

SENDER_CODE = """import socket, time, os, sys
dur=float(sys.argv[1]); ip=sys.argv[2]; port=int(sys.argv[3])
c=socket.socket(); c.connect((ip,port))
buf=os.urandom(65536); t0=time.time()
while time.time()-t0<dur:
    try: c.sendall(buf)
    except: break
"""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-py", required=True)
    ap.add_argument("--matrix-var", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--container-prefix", type=str, default="mn.worker")
    ap.add_argument("--ip-prefix", type=str, default="10.0.0.")
    ap.add_argument("--port-base", type=int, default=9100)
    ap.add_argument("--tolerance", type=float, default=0.35)
    args = ap.parse_args()

    B = load_matrix_from_py(args.matrix_py, args.matrix_var)
    n = args.n if args.n is not None else len(B)
    if len(B) < n or any(len(row) < n for row in B):
        raise ValueError(f"Matrix smaller than n={n}")

    random.seed(args.seed)

    pairs: List[Tuple[int,int]] = []
    used = set()
    while len(pairs) < args.pairs:
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j or (i, j) in used:
            continue
        used.add((i, j))
        pairs.append((i, j))

    print(f"# Sanity check (extracted): {args.matrix_py}:{args.matrix_var}, n={n}, pairs={len(pairs)}, duration={args.duration}s")
    print("# src dst expected_Mbps measured_Mbps rel_err verdict")

    ok = 0
    for k, (i, j) in enumerate(pairs):
        src = f"{args.container_prefix}{i}"
        dst = f"{args.container_prefix}{j}"
        dst_ip = overlay_ip(args.ip_prefix, j)
        port = args.port_base + k

        recv = subprocess.Popen(
            ["docker", "exec", dst, "python3", "-c", RECEIVER_CODE, str(args.duration), str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(0.2)

        docker_exec_py(src, SENDER_CODE, [str(args.duration), dst_ip, str(port)], timeout=int(args.duration)+5)

        out, err = recv.communicate(timeout=int(args.duration)+5)
        out = ANSI_RE.sub("", (out or "")).strip()
        if not out:
            raise RuntimeError(f"Receiver produced no output for {i}->{j}. stderr={err!r}")
        measured_mbps = float(json.loads(out)["mbps"])

        expected = float(B[i][j])
        rel = abs(measured_mbps - expected) / expected if expected > 0 else 0.0
        verdict = "OK" if rel <= args.tolerance else "FAIL"
        ok += (verdict == "OK")

        print(f"{i:>3} {j:>3} {expected:>10.3f} {measured_mbps:>12.3f} {rel:>7.2f} {verdict}")

    print(f"# Summary: {ok}/{len(pairs)} OK (tolerance={args.tolerance:.2f})")
    return 0 if ok == len(pairs) else 1

if __name__ == "__main__":
    raise SystemExit(main())
