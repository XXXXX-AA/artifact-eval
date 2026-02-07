
import logging
from mpi4py import MPI

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

def FedML_ADPSGD(process_id, worker_number, device, comm,
                 model, train_data_num, train_data_global, test_data_global,
                 train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, model_trainer=None):
    rank = process_id

    if model_trainer is None:
        model_trainer = MyModelTrainer(model, device, args)
    model_trainer.set_id(process_id)
    # model_trainer = MyModelTrainer(model, device, args)

    # topology: 2-out-degree symmetric graph by default (can be tuned via args.degree)
    degree = int(getattr(args, "graph_degree", 2))
    tpmgr = SymmetricTopologyManager(worker_number, degree)
    tpmgr.generate_topology()

    timer = Timer(
        verbosity_level=1 if track_time else 0,
        log_fn=Logger.log_timer
    )
    metrics = Metrics([1], task=args.task)
    
    worker = DecentralizedWorker(rank, tpmgr, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number,
                 device, model, args, model_trainer, timer, metrics)

    manager = DecentralizedWorkerManager(args, comm, process_id, worker_number, worker, tpmgr,
                                         model_trainer, timer, metrics)
    manager.run()
