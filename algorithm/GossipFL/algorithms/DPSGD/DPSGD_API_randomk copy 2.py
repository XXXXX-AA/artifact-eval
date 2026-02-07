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


class OverlayMaskedSymmetricTopologyManager(SymmetricTopologyManager):
    def __init__(self, n, neighbor_num=2, overlay_mask=None, rng=None):
        super().__init__(n, neighbor_num)
        self.overlay_mask = overlay_mask
        self._rng = rng if rng is not None else np.random.default_rng()

    def generate_topology(self):
        if self.overlay_mask is None:
            return super().generate_topology()

        n = self.n
        allowed = self.overlay_mask.astype(bool)
        np.fill_diagonal(allowed, True)

        topo = np.zeros((n, n), dtype=np.float32)

        # Ring-like edges: prefer i -> i+1 if allowed, otherwise choose any allowed neighbor
        for i in range(n):
            j = (i + 1) % n
            if i != j and allowed[i, j]:
                topo[i, j] = 1.0
                if allowed[j, i]:
                    topo[j, i] = 1.0
            else:
                candidates = np.where(allowed[i])[0]
                candidates = candidates[candidates != i]
                if candidates.size > 0:
                    j = self._rng.choice(candidates)
                    topo[i, j] = 1.0
                    if allowed[j, i]:
                        topo[j, i] = 1.0

        # Random links (symmetric when possible), restricted to allowed edges
        k = int(self.neighbor_num)
        for i in range(n):
            candidates = np.where(allowed[i])[0]
            candidates = candidates[candidates != i]
            candidates = [j for j in candidates if topo[i, j] == 0]
            if len(candidates) == 0:
                continue
            choose = self._rng.choice(candidates, size=min(k, len(candidates)), replace=False)
            for j in np.atleast_1d(choose):
                topo[i, j] = 1.0
                if allowed[j, i]:
                    topo[j, i] = 1.0

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
    tpmgr = OverlayMaskedSymmetricTopologyManager(worker_number, 1, overlay_mask=mask)
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
