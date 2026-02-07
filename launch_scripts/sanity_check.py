#!/usr/bin/env python3
"""sanity_check.py

Sanity-check a Containernet/Mininet bandwidth matrix enforced by Linux tc (HTB + netem).

Per sampled pair i -> j:
  1) Start a receiver in container j (prints JSON: bytes, mbps).
  2) Start a sender in container i for D seconds.
  3) Report expected Mbps (matrix) vs measured Mbps (receiver-side), plus optional tc configured rate.

Assumptions (override via CLI flags):
  - Containers: mn.worker0, mn.worker1, ...
  - Overlay IPs: 10.0.0.(idx+1)
  - Data-plane interface: worker{idx}-eth0
  - tc shaping already installed on sender egress interface.

Notes:
  - Receiver-side throughput is the correct "effective bandwidth" to compare against B[i][j].
  - Sender-side send() rates can exceed receiver due to TCP buffering and timing mismatch.
"""

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import time
from typing import List, Optional, Tuple

ANSI_RE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

def _run(cmd: List[str], timeout: Optional[int] = None, check: bool = False, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        timeout=timeout,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )

def docker_exec(container: str, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return _run(["docker", "exec", container] + args, timeout=timeout)

def docker_exec_py(container: str, py_code: str, py_args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return docker_exec(container, ["python3", "-c", py_code] + py_args, timeout=timeout)

def load_matrix_from_py(path: str, var_name: str) -> List[List[float]]:
    """Parse a python file and literal-eval the assigned list for var_name, without executing the file."""
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    val = ast.literal_eval(node.value)
                    if not isinstance(val, list) or not val or not isinstance(val[0], list):
                        raise ValueError(f"{var_name} is not a 2D list")
                    return [[float(x) for x in row] for row in val]
    raise ValueError(f"Could not find assignment to {var_name} in {path}")

def overlay_ip(ip_prefix: str, idx: int) -> str:
    return f"{ip_prefix}{idx+1}"

def guess_tc_rate_mbps(container: str, intf: str, dst_ip: str) -> Optional[float]:
    """Best-effort: dst_ip -> flowid via tc filter, then parse class rate. Returns Mbps or None."""
    cp_f = docker_exec(container, ["bash", "-lc", f"tc filter show dev {intf} parent 1: 2>/dev/null || true"])
    out = (cp_f.stdout or "") + "\n" + (cp_f.stderr or "")
    out = ANSI_RE.sub("", out)

    flowid = None

    m = re.search(rf"match ip dst\s+{re.escape(dst_ip)}/\d+\s+flowid\s+(1:\d+)", out)
    if m:
        flowid = m.group(1)

    if flowid is None:
        parts = dst_ip.split(".")
        if len(parts) == 4:
            try:
                hx = "".join(f"{int(p):02x}" for p in parts)
                m2 = re.search(rf"match\s+{hx}/[0-9a-fA-F]{{8}}.*?flowid\s+(1:\d+)", out)
                if m2:
                    flowid = m2.group(1)
            except Exception:
                pass

    if flowid is None:
        return None

    class_minor = flowid.split(":")[1]

    cp_c = docker_exec(container, ["bash", "-lc", f"tc class show dev {intf} 2>/dev/null | grep ' 1:{class_minor} ' || true"])
    line = ANSI_RE.sub("", (cp_c.stdout or "")).strip()
    if not line:
        return None

    mrate = re.search(r"rate\s+([0-9.]+)([KMG]?bit)", line, re.IGNORECASE)
    if not mrate:
        return None
    val = float(mrate.group(1))
    unit = mrate.group(2).lower()
    if unit == "kbit":
        return val / 1000.0
    if unit == "mbit":
        return val
    if unit == "gbit":
        return val * 1000.0
    return None

RECEIVER_CODE = """    import socket, time, json, sys
dur=float(sys.argv[1]); port=int(sys.argv[2])
s=socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((\"0.0.0.0\", port))
s.listen(1)
c,addr=s.accept()
c.settimeout(1.0)
t0=time.time(); n=0
while True:
    if time.time()-t0>=dur: break
    try:
        data=c.recv(65536)
    except socket.timeout:
        continue
    if not data: break
    n+=len(data)
res={\"bytes\":n,\"duration\":dur,\"mbps\":(n*8.0/dur/1e6)}
print(json.dumps(res))
"""

SENDER_CODE = """    import socket, time, os, sys
dur=float(sys.argv[1]); host=sys.argv[2]; port=int(sys.argv[3])
c=socket.socket()
c.connect((host, port))
buf=os.urandom(65536)
t0=time.time(); sent=0
while True:
    if time.time()-t0>=dur: break
    try:
        c.sendall(buf)
        sent += len(buf)
    except (BrokenPipeError, ConnectionResetError):
        break
print(sent)
"""

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-py", type=str, required=True, help="Path to a .py file that defines the bandwidth matrix as a literal 2D list")
    ap.add_argument("--matrix-var", type=str, required=True, help="Variable name of the matrix inside the python file, e.g., bandwidth_list15")
    ap.add_argument("--n", type=int, default=None, help="Number of workers; defaults to matrix size")
    ap.add_argument("--pairs", type=int, default=8, help="How many random pairs to test")
    ap.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--container-prefix", type=str, default="mn.worker", help="Container name prefix (default: mn.worker)")
    ap.add_argument("--ip-prefix", type=str, default="10.0.0.", help="Overlay IP prefix (default: 10.0.0.)")
    ap.add_argument("--intf-template", type=str, default="worker{idx}-eth0", help="Data-plane interface template inside container")
    ap.add_argument("--port-base", type=int, default=9100, help="Base TCP port for tests (each pair uses port_base + k)")
    ap.add_argument("--tolerance", type=float, default=0.35, help="Allowed relative error vs expected Mbps (receiver-side)")
    args = ap.parse_args()

    B = load_matrix_from_py(args.matrix_py, args.matrix_var)
    n = args.n if args.n is not None else len(B)
    if len(B) < n or any(len(row) < n for row in B):
        raise ValueError(f"Matrix is smaller than n={n}")

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

    print(f"# Sanity check: matrix={args.matrix_py}:{args.matrix_var}, n={n}, pairs={len(pairs)}, duration={args.duration}s")
    print("# Columns: src dst expected_Mbps measured_Mbps rel_err tc_rate_Mbps(optional) verdict")

    ok_cnt = 0

    for k, (i, j) in enumerate(pairs):
        src = f"{args.container_prefix}{i}"
        dst = f"{args.container_prefix}{j}"
        dst_ip = overlay_ip(args.ip_prefix, j)
        port = args.port_base + k

        recv_proc = subprocess.Popen(
            ["docker", "exec", dst, "python3", "-c", RECEIVER_CODE, str(args.duration), str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.2)

        docker_exec_py(src, SENDER_CODE, [str(args.duration), dst_ip, str(port)], timeout=int(args.duration)+10)

        try:
            out, err = recv_proc.communicate(timeout=int(args.duration)+10)
        except subprocess.TimeoutExpired:
            recv_proc.kill()
            out, err = recv_proc.communicate()

        out = ANSI_RE.sub("", (out or "")).strip()
        err = ANSI_RE.sub("", (err or "")).strip()

        measured_mbps = None
        if out:
            try:
                measured_mbps = float(json.loads(out)["mbps"])
            except Exception:
                measured_mbps = None

        expected_mbps = float(B[i][j])
        rel_err = None
        verdict = "FAIL"
        if measured_mbps is not None and expected_mbps > 0:
            rel_err = abs(measured_mbps - expected_mbps) / expected_mbps
            verdict = "OK" if rel_err <= args.tolerance else "FAIL"

        intf = args.intf_template.format(idx=i)
        tc_rate = guess_tc_rate_mbps(src, intf, dst_ip)

        if verdict == "OK":
            ok_cnt += 1

        rel_err_s = f"{rel_err:.2f}" if rel_err is not None else "NA"
        meas_s = f"{measured_mbps:.3f}" if measured_mbps is not None else "NA"
        tc_s = f"{tc_rate:.3f}" if tc_rate is not None else "NA"

        print(f"{i:>3} {j:>3} {expected_mbps:>10.3f} {meas_s:>12} {rel_err_s:>7} {tc_s:>10} {verdict}")

        if measured_mbps is None:
            print(f"  [warn] receiver parse failed. raw_out={out!r} raw_err={err!r}")

    print(f"# Summary: {ok_cnt}/{len(pairs)} within tolerance={args.tolerance:.2f}")
    return 0 if ok_cnt == len(pairs) else 1

if __name__ == "__main__":
    raise SystemExit(main())
