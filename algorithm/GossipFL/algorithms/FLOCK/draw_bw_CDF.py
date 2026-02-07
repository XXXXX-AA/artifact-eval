# This cell generates CDF plots for bandwidth matrices for 14, 32, and 64 nodes,
# attempting to use the user's bandwidth-generation code in /mnt/data/utils.py.
# It will fall back to random matrices if "real" presets are unavailable.
import importlib.util
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import types

# Try importing the user's utils.py dynamically
utils_path = Path("/mnt/data/utils.py")
if utils_path.exists():
    spec = importlib.util.spec_from_file_location("user_utils", str(utils_path))
    user_utils = importlib.util.module_from_spec(spec)
    sys.modules["user_utils"] = user_utils
    spec.loader.exec_module(user_utils)
else:
    raise FileNotFoundError("utils.py not found at /mnt/data/utils.py")

# Helper: create args-like object for generate_bandwidth
class Args:
    def __init__(self, n, bw_type="real"):
        self.client_num_in_total = n
        self.bandwidth_type = bw_type

def _to_numpy_bandwidth(bw):
    """Convert different possible return types to a clean numpy array."""
    if isinstance(bw, np.ndarray):
        arr = bw.copy()
    else:
        arr = np.array(bw, dtype=float)
    return arr

def _is_numeric_matrix(arr: np.ndarray) -> bool:
    try:
        # Ensure it's 2D numeric and no NaN/inf and no non-finite
        return arr.ndim == 2 and np.isfinite(arr).all()
    except Exception:
        return False

def _upper_triangle_values(arr: np.ndarray) -> np.ndarray:
    """Extract the strict upper-triangular values (i<j) excluding zeros on diagonal."""
    n = arr.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    vals = arr[triu_idx]
    # Filter out non-positive or zeros if any (optional; we keep >0)
    vals = vals[np.isfinite(vals)]
    return vals

def _ecdf(values: np.ndarray):
    """Return x,y for empirical CDF of values."""
    v = np.sort(values)
    n = len(v)
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n+1) / n
    return v, y

def _get_bandwidth_matrix(n: int):
    """Try to get user's 'real' bandwidth; fall back to random."""
    # First try 'real'
    try:
        bw = user_utils.generate_bandwidth(Args(n, "real"))
        arr = _to_numpy_bandwidth(bw)
        if _is_numeric_matrix(arr) and arr.shape == (n, n):
            return arr, "real"
    except Exception as e:
        pass
    # Fall back to 'random'
    try:
        bw = user_utils.generate_bandwidth(Args(n, "random"))
        arr = _to_numpy_bandwidth(bw)
        if _is_numeric_matrix(arr) and arr.shape == (n, n):
            return arr, "random"
    except Exception as e:
        pass
    # If both fail, synthesize a symmetric random matrix
    rng = np.random.default_rng(7)
    base = rng.uniform(0.5, 200.0, size=(n, n))
    arr = np.triu(base, 1)
    arr = arr + arr.T
    np.fill_diagonal(arr, 0.0)
    return arr, "synthetic"

def plot_cdf_for_n(n: int, save_dir: Path):
    arr, src = _get_bandwidth_matrix(n)
    vals = _upper_triangle_values(arr)
    x, y = _ecdf(vals)
    # Create plot
    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Link bandwidth")
    plt.ylabel("CDF")
    plt.title(f"Bandwidth CDF — {n} nodes ({src})")
    out = save_dir / f"bandwidth_cdf_{n}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out, src, vals

save_dir = Path(".")
outputs = []
meta = []
for n in (14, 32, 64):
    out, src, vals = plot_cdf_for_n(n, save_dir)
    outputs.append(out)
    meta.append((n, src, len(vals), float(np.min(vals)) if len(vals) else float("nan"),
                 float(np.percentile(vals, 50)) if len(vals) else float("nan"),
                 float(np.percentile(vals, 90)) if len(vals) else float("nan"),
                 float(np.max(vals)) if len(vals) else float("nan")))

# Report file locations and a small summary
outputs, meta
