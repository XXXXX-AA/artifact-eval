import logging
from mpi4py import MPI
import os
import time
from pathlib import Path
import numpy as np

from fedml_core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager

from utils.timer_with_cuda import Timer
from utils.metrics import Metrics
from utils.logger import Logger


from .decentralized_worker import DecentralizedWorker
from .decentralized_worker_manager import DecentralizedWorkerManager
from .MyModelTrainer import MyModelTrainer


track_time = True


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        self.count -= 1
        return True


class OverlayMaskedSymmetricTopologyManager(SymmetricTopologyManager):
    def __init__(self, n, neighbor_num=2, overlay_mask=None, rng=None):
        super().__init__(n, neighbor_num)
        self.overlay_mask = overlay_mask
        self._rng = rng if rng is not None else np.random.default_rng()

    def _is_connected(self, adj: np.ndarray) -> bool:
        n = adj.shape[0]
        if n == 0:
            return True
        seen = np.zeros(n, dtype=bool)
        stack = [0]
        seen[0] = True
        while stack:
            i = stack.pop()
            nbrs = np.where(adj[i] > 0)[0]
            for j in nbrs:
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        return seen.all()

    def _build_adj_with_degree_bounds(self, allowed: np.ndarray, min_deg: int, max_deg: int):
        n = allowed.shape[0]
        adj = np.zeros((n, n), dtype=np.int8)
        deg = np.zeros(n, dtype=np.int32)

        # Build a connected backbone with degree cap.
        edges = np.argwhere(np.triu(allowed, 1))
        self._rng.shuffle(edges)
        uf = _UnionFind(n)
        for i, j in edges:
            if uf.find(i) != uf.find(j) and deg[i] < max_deg and deg[j] < max_deg:
                adj[i, j] = 1
                adj[j, i] = 1
                deg[i] += 1
                deg[j] += 1
                uf.union(i, j)
                if uf.count == 1:
                    break
        if uf.count != 1:
            return None

        # Raise low-degree nodes to min_deg without violating max_deg.
        max_iters = n * max_deg * 4
        for _ in range(max_iters):
            low = np.where(deg < min_deg)[0]
            if low.size == 0:
                break
            progressed = False
            self._rng.shuffle(low)
            for i in low:
                candidates = np.where(allowed[i])[0]
                candidates = candidates[candidates != i]
                candidates = candidates[adj[i, candidates] == 0]
                candidates = candidates[deg[candidates] < max_deg]
                if candidates.size == 0:
                    continue
                # Prefer the lowest-degree neighbor to balance degrees.
                j = candidates[np.argmin(deg[candidates])]
                adj[i, j] = 1
                adj[j, i] = 1
                deg[i] += 1
                deg[j] += 1
                progressed = True
            if not progressed:
                break

        if np.any(deg < min_deg):
            return None
        if not self._is_connected(adj):
            return None
        return adj

    def generate_topology(self):
        if self.overlay_mask is None:
            return super().generate_topology()

        n = self.n
        allowed = self.overlay_mask.astype(bool)
        np.fill_diagonal(allowed, False)

        min_deg = 2 if n > 2 else max(0, n - 1)
        max_candidates = [2, 3, 4]
        max_candidates = [min(m, max(0, n - 1)) for m in max_candidates]
        max_candidates = [m for m in max_candidates if m >= min_deg]
        max_candidates = sorted(set(max_candidates))
        adj = None
        chosen_max = None
        for max_deg in max_candidates:
            for _ in range(200):
                adj_try = self._build_adj_with_degree_bounds(allowed, min_deg, max_deg)
                if adj_try is not None:
                    adj = adj_try
                    chosen_max = max_deg
                    break
            if adj is not None:
                break

        if adj is None:
            # Fallback: build a connected graph with a relaxed degree cap.
            for max_deg in reversed(max_candidates):
                for _ in range(200):
                    adj_try = self._build_adj_with_degree_bounds(allowed, 1, max_deg)
                    if adj_try is not None:
                        adj = adj_try
                        chosen_max = max_deg
                        break
                if adj is not None:
                    break

        if adj is None:
            adj = np.zeros((n, n), dtype=np.int8)
            chosen_max = max_candidates[-1]

        if chosen_max is not None:
            logging.info(f"randomk topology degree target min={min_deg} max={chosen_max}")

        topo = adj.astype(np.float32)
        np.fill_diagonal(topo, 1.0)
        row_sum = topo.sum(axis=1, keepdims=True)
        zero_rows = (row_sum.squeeze() == 0)
        if np.any(zero_rows):
            for i in np.where(zero_rows)[0]:
                topo[i, :] = 0.0
                topo[i, i] = 1.0
            row_sum = topo.sum(axis=1, keepdims=True)
        topo = topo / row_sum
        self.topology = topo


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_DPSGD(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer=None):
    # args = Args(job)
    # initialize the topology (ring)
    if model_trainer is None:
        model_trainer = MyModelTrainer(model, device, args)
    model_trainer.set_id(process_id)

    # configure logger.
    # conf.logger = logging.Logger(conf.checkpoint_dir)

    # configure timer.
    timer = Timer(
        verbosity_level=1 if track_time else 0,
        log_fn=Logger.log_timer
    )
    metrics = Metrics([1], task=args.task)

    mask = _overlay_mask(worker_number)
    tpmgr = OverlayMaskedSymmetricTopologyManager(worker_number, 2, overlay_mask=mask)
    current_dir = Path(__file__).parent
    out_dir = current_dir / "dpsgd-generate_bandwidth"
    topo_path = out_dir / "topology.npy"
    ready_path = out_dir / "READY"

    if process_id == 0:
        tpmgr.generate_topology()
        topo = np.asarray(tpmgr.topology, dtype=np.float32)
        logging.info(f"Generated topology:\n{tpmgr.topology}")

        adj = (topo > 0).astype(np.int32)
        logging.info(f"adj symmetric? {np.array_equal(adj, adj.T)}")
        logging.info(f"weight symmetric? {np.allclose(topo, topo.T)}")
        deg = adj.sum(axis=1) - 1
        logging.info(f"deg min={deg.min()} max={deg.max()} isolated={(deg==0).sum()}")
        _atomic_save_npy(topo_path, topo)

        # NOTE: comment translated from Chinese
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_tmp = ready_path.with_name(ready_path.name + ".tmp")
        with open(ready_tmp, "w") as f:
            f.write("ok")
            f.flush()
            os.fsync(f.fileno())
        os.replace(ready_tmp, ready_path)
        dfd = os.open(str(ready_path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    else:
        # NOTE: comment translated from Chinese
        for _ in range(240):
            if ready_path.exists():
                break
            time.sleep(0.25)
        topo = _wait_and_load_npy(topo_path)
        tpmgr.topology = topo
    # logging.info(tpmgr.topology)

    # initialize the decentralized trainer (worker)
    worker_index = process_id
    worker = DecentralizedWorker(worker_index, tpmgr, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number, 
                 device, model, args, model_trainer, timer, metrics)

    client_manager = DecentralizedWorkerManager(args, comm, process_id, worker_number, worker, tpmgr, model_trainer,
                                                timer, metrics)
    client_manager.run()


def _overlay_mask(worker_number: int) -> np.ndarray:
    from randomk_overlay_matrix import bandwidth_list64
    bw = np.array(bandwidth_list64, dtype=np.float32)
    if worker_number > bw.shape[0]:
        raise ValueError(f"worker_number ({worker_number}) exceeds overlay size ({bw.shape[0]})")
    if worker_number < bw.shape[0]:
        start = bw.shape[0] - worker_number
        bw = bw[start:bw.shape[0], start:bw.shape[0]]
    allowed = (bw > 0)
    allowed = allowed & allowed.T
    mask = allowed.astype(np.float32)
    np.fill_diagonal(mask, 1.0)
    return mask


def _fsync_file_and_dir(path: Path):
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def _atomic_save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")  # NOTE: comment translated from Chinese
    # NOTE: comment translated from Chinese
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # NOTE: comment translated from Chinese
    dfd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _wait_and_load_npy(path: Path, retry=240, interval=0.25) -> np.ndarray:
    path = Path(path)
    for _ in range(retry):
        try:
            if path.exists() and path.stat().st_size > 0:
                return np.load(path, allow_pickle=False)
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"topology file not found or empty: {path}")
