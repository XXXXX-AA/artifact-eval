import logging

from mpi4py import MPI

import os, time
from pathlib import Path
import numpy as np

from utils.timer_with_cuda import Timer
from utils.metrics import Metrics
from utils.logger import Logger
from fedml_core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager

from .decentralized_worker import DecentralizedWorker
from .decentralized_worker_manager import DecentralizedWorkerManager
from .MyModelTrainer import MyModelTrainer


track_time = True



def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_CHOCO_SGD(process_id, worker_number, device, comm, model, train_data_num, train_data_global, test_data_global,
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

    tpmgr = SymmetricTopologyManager(worker_number, 2)
    """
    tpmgr: SymmetricTopologyManager(worker_number, neighbor_num)
    rank: 当前 worker 索引（0..world_size-1）
    world_size: 节点数
    shared_dir: 共享卷上的绝对路径，如 /shared/gossip   （必须是共享卷！）
    tag: 区分不同实验/轮次的标签（可选）
    """
    current_dir = Path(__file__).parent
    out_dir = current_dir / "choco-sgd-generate_bandwidth"
    topo_path = out_dir / "topology.npy"
    ready_path = out_dir / "READY"

    if process_id == 0:
        tpmgr.generate_topology()
        topo = np.asarray(tpmgr.topology, dtype=np.float32)
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