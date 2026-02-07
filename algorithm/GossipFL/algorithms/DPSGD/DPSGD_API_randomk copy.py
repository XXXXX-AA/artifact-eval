import logging
from mpi4py import MPI
import numpy as np

from fedml_core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager

from utils.timer_with_cuda import Timer
from utils.metrics import Metrics
from utils.logger import Logger


from .decentralized_worker import DecentralizedWorker
from .decentralized_worker_manager import DecentralizedWorkerManager
from .MyModelTrainer import MyModelTrainer


track_time = True


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

    tpmgr = SymmetricTopologyManager(worker_number, 2)
    topo = _build_overlay_topology(worker_number)
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


def _build_overlay_topology(worker_number: int):
    from randomk_overlay_matrix import bandwidth_list64
    bw = np.array(bandwidth_list64, dtype=np.float32)
    if worker_number > bw.shape[0]:
        raise ValueError(f"worker_number ({worker_number}) exceeds overlay size ({bw.shape[0]})")
    if worker_number < bw.shape[0]:
        start = bw.shape[0] - worker_number
        bw = bw[start:bw.shape[0], start:bw.shape[0]]
    topo = (bw > 0).astype(np.float32)
    np.fill_diagonal(topo, 1.0)
    row_sum = topo.sum(axis=1, keepdims=True)
    topo = topo / row_sum
    return topo
