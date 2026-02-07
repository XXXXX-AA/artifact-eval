#!/usr/bin/env python3
"""
sanity_check_v3.py

Sanity-check a Containernet/Mininet bandwidth matrix enforced by Linux tc (HTB + netem).

Usage:
  python3 sanity_check_v3.py \
    --matrix-py /path/to/matrices.py \
    --matrix-var bandwidth_list15 \
    --n 15 \
    --pairs 10 \
    --duration 5
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

def run(cmd, timeout=None):
    return subprocess.run(
        cmd, timeout=timeout, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

def docker_exec(container: str, args: List[str], timeout=None):
    return run(["docker", "exec", container] + args, timeout)

def docker_exec_py(container: str, code: str, args: List[str], timeout=None):
    return docker_exec(container, ["python3", "-c", code] + args, timeout)

def load_matrix_from_py(path: str, var_name: str) -> List[List[float]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=path)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    val = ast.literal_eval(node.value)
                    return [[float(x) for x in row] for row in val]

    raise ValueError(f"Cannot find matrix variable {var_name} in {path}")

def overlay_ip(idx: int) -> str:
    return f"10.0.0.{idx+1}"

RECEIVER = """
import socket, time, json, sys
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
print(json.dumps({"mbps": n*8/dur/1e6}))
"""

SENDER = """
import socket, time, os, sys
dur=float(sys.argv[1]); ip=sys.argv[2]; port=int(sys.argv[3])
c=socket.socket(); c.connect((ip,port))
buf=os.urandom(65536); t0=time.time()
while time.time()-t0<dur:
    try: c.sendall(buf)
    except: break
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-py", required=True)
    ap.add_argument("--matrix-var", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    B = load_matrix_from_py(args.matrix_py, args.matrix_var)
    n = args.n or len(B)
    random.seed(args.seed)

    pairs = set()
    while len(pairs) < args.pairs:
        i,j = random.randrange(n), random.randrange(n)
        if i!=j: pairs.add((i,j))

    print("# src dst expected_Mbps measured_Mbps rel_err verdict")

    ok = 0
    for k,(i,j) in enumerate(pairs):
        src=f"mn.worker{i}"
        dst=f"mn.worker{j}"
        ip=overlay_ip(j)
        port=9100+k

        recv = subprocess.Popen(
            ["docker","exec",dst,"python3","-c",RECEIVER,str(args.duration),str(port)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(0.2)

        docker_exec_py(src, SENDER, [str(args.duration), ip, str(port)],
                       timeout=int(args.duration)+5)

        out,_ = recv.communicate(timeout=int(args.duration)+5)
        out = ANSI_RE.sub("", out).strip()
        mbps = json.loads(out)["mbps"]

        exp = B[i][j]
        rel = abs(mbps-exp)/exp if exp>0 else 0
        verdict = "OK" if rel<=0.35 else "FAIL"
        ok += (verdict=="OK")

        print(f"{i:>3} {j:>3} {exp:>10.3f} {mbps:>12.3f} {rel:>7.2f} {verdict}")

    print(f"# Summary: {ok}/{len(pairs)} OK")

if __name__ == "__main__":
    main()
