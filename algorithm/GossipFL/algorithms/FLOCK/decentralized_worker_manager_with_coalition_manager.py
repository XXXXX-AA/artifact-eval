import logging
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from copy import deepcopy
import torch, gc, os, psutil
proc = psutil.Process(os.getpid())
def mem_mb():
    rss = proc.memory_info().rss / (1024*1024)
    cuda = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0
    return rss, cuda

def cpu_mem_str(tag=""):
    try:
        cpu = proc.cpu_percent(interval=None)  
    except Exception:
        cpu = -1.0
    rss, cuda = mem_mb()
    return f"{tag} CPU={cpu:.1f}% RSS={rss:.1f}MB CUDA={cuda:.1f}MB"

import traceback
from mpi4py import MPI
import numpy as np

from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message

from algorithms.baseDecent.decentralized_worker_manager_FLOCK import BaseDecentralizedWorkerManager
from algorithms.FLOCK.snapshot_writer import SnapshotWriter
from algorithms.FLOCK.coalition_manager import CoalitionManager


from utils.context import (
    raise_MPI_error,
    raise_error_without_process,
    get_lock,
)
from utils.timer import Timer
from utils.tracker import RuntimeTracker
# from fedml_api.utils.timer_with_cuda import Timer
from utils.metrics import Metrics
from utils.wandb_util import wandb_log
from utils.data_utils import (
    get_data,
    apply_gradient
)
from utils.tensor_buffer import (
    TensorBuffer
)

from .compressor import FLOCK_FLCompressor
from .utils import generate_bandwidth

# from .FLOCK_topology_manager import FLOCKTopologyManager
from .message_define import MyMessage

from queue import Queue, Empty, Full

comm = MPI.COMM_WORLD

class DecentralizedWorkerManager(BaseDecentralizedWorkerManager):
    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics):
        super().__init__(args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics)
        self.neighbor_transfer_lock = threading.Lock()
        self.sync_receive_all_event = threading.Event()
        self.complete_aggregation_event = threading.Event()

        
        self.msg_received = 0
        self._rx_queue: Queue = Queue(maxsize=3) #2
        self._rx_stop = threading.Event()
        self._rx_thread = threading.Thread(
            name=f"rx-io-{rank}", target=self._rx_loop, daemon=True
        )
        
        self.training_thread = threading.Thread(
            name="training", target=self.run_sync, daemon=True
        )


        # ==== NEW: CPU decode thread-pool + decoded queue (size=3) + tokens ====
        # self._decode_pool = ThreadPoolExecutor(max_workers=min(8, max(2, (os.cpu_count() or 4)//2)))
        # self._decoded_queue = Queue(maxsize=3)
        # self._decode_tokens = threading.BoundedSemaphore(value=3)
        # self._agg_stop = threading.Event()
        # self._agg_thread = threading.Thread(
        #     name=f"agg-io-{rank}", target=self._agg_loop, daemon=True
        # )
        
        # # for containernet
        # self._decode_pool = ThreadPoolExecutor(max_workers=2) # for containernet
        # self._decoded_queue = Queue(maxsize=2)
        # self._decode_tokens = threading.BoundedSemaphore(value=2)
        # self._agg_stop = threading.Event()
        # self._agg_thread = threading.Thread(
        #     name=f"agg-io-{rank}", target=self._agg_loop, daemon=True
        # )
        # for containernet
        self._decode_pool = ThreadPoolExecutor(max_workers=3) # for containernet
        self._decoded_queue = Queue(maxsize=3)
        self._decode_tokens = threading.BoundedSemaphore(value=3)
        self._agg_stop = threading.Event()
        self._agg_thread = threading.Thread(
            name=f"agg-io-{rank}", target=self._agg_loop, daemon=True
        )


        # whether all workers finished
        self.ifStop = False
        self.round = 0  # Current round index

        
        self.bandwidth = generate_bandwidth(args)  
        # if self.worker_index == 0:
        # logging.info(f"[rank {self.worker_index}] Bandwidth matrix:\n{self.bandwidth}")
        bw_raw = self.bandwidth[self.worker_index]     # 1-D
        bw_log_max = np.log1p(bw_raw).max() or 1.0  
        self.bw_norm = np.log1p(bw_raw) / bw_log_max   # 0-1
        self.outs = self.my_get_out_neighbor_idx_list(self.worker_index)
        
        self.coalition_manager = CoalitionManager(args, self.worker_index, size, self.bandwidth, self.bw_norm, self.outs, self.epochs, self.worker.num_iterations)
        self.coalition_manager.set_active_coalition_join_callback(self.send_active_coalition_join_notification)
        
        
        # # if self.worker_index == 0:
        # # logging.info(f"[rank {self.worker_index}] Bandwidth matrix:\n{self.bandwidth}")
        # bw_raw = self.bandwidth[self.worker_index]     # 1-D
        
        # self.bw_norm = np.log1p(bw_raw) / bw_log_max   # 0-1
        # self.outs = self.my_get_out_neighbor_idx_list(self.worker_index)
        
        
        # self.util_eps = 0.3                            # ε‑greedy
        # self.lambda_bw0 = 3.0
        # self.lambda_sim0 = 0.8
        # self.lambda_dist = 0.05
        # self.time_const = 1.0 * self.epochs
        
        # self.util_threshold = 0.0
        # self.target_good_ratio = 0.3
        # self.tolerance_ratio = 0.05
        # self.adjust_step = 0.002
        # self.time_window = int(np.ceil(3 * np.log(size)))
        # self.last_chosen_round = np.zeros(size, dtype=int)

        
        # self.good_set = set()
        # self.bad_set = set(range(size))

        
        

        # compression part
        self.compression = args.compression
        assert self.compression in ["topk", "randomk", "quantize", "sign"]
        self.compressor = FLOCK_FLCompressor(comm_op=self.compression,
                                            compress_ratio=args.compress_ratio,  # args.compress_ratio
                                            quantize_level=args.quantize_level,
                                            is_biased=(args.is_biased == 1),)

        
        self.worker.set_rx_compressor(self.compressor)
        self.worker.set_eval_callback(self._on_neighbor_eval)
        self.finish_round = self.epochs * self.worker.num_iterations-1

        # self.snapshot_writer = SnapshotWriter(f"./param_snapshots/{self.worker_index}",1)

    def run(self):
        logging.debug("Wait for the barrier!")
        comm.Barrier()
        time.sleep(1)
        logging.debug("MPI exit barrier!")

        proc.cpu_percent(interval=None)

        self._rx_thread.start()
        self.training_thread.start()
        self._agg_thread.start()

        if self.worker_index == 0:
            logging.debug("COORDINATOR notify clients to start!")
            self.coodinator_thread.start()
            self.notify_clients()
        super().run()

    def register_message_receive_handlers(self):
        
        # self.register_message_receive_handler(MyMessage.MSG_TYPE_CLUSTER_INFO,
        #                                       self.handle_msg_cluster_info)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR,
                                              self.handle_msg_from_neighbor)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR,
                                              self.handle_msg_client_to_coordinator)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_COORDINATOR_TO_CLIENT,
                                              self.handle_msg_coordinator_to_client)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_COAL_INFO,
                                              self.handle_msg_with_coalition_info)

    def handle_msg_client_to_coordinator(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.debug("handle_msg_client_to_coordinator.  Sender_id = " + str(sender_id)+" Finished training.")

        with get_lock(self.total_metric_lock):
            logging.debug("get metric lock, handle_msg_client_to_coordinator. sender_id = " + str(sender_id)+" Finished training.")
            self.flag_client_finish_dict[sender_id] = True
            self.check_worker_finish_and_notify()

    def handle_msg_coordinator_to_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        logging.debug("handle_msg_coordinator_to_client. Sender_id = " + str(sender_id)+" Finished training.")      

    def handle_msg_from_neighbor(self, msg_params):
       """改为入队，真正的落盘交给 _rx_loop；避免阻塞训练线程。"""
       try:
        #    self._rx_queue.put_nowait(msg_params)
        # self._rx_queue.put(msg_params, block=True, timeout=0.5)
        self._rx_queue.put(msg_params, block=True)
       except Full:
           
           try:
               sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
               sender_round = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_ROUND)
               logging.warning("rx queue full; process inline for sender=%s round=%s", sender_id, sender_round)
               with get_lock(self.neighbor_transfer_lock):
                   self.worker.add_result_for_flock(sender_id, sender_round, msg_params)
           except Exception:
               logging.exception("fallback add_result_for_flock failed")
       
       return                                    

    def handle_msg_with_coalition_info(self, msg_params):
        try:
            sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
            # logging.info(f"handle_msg_with_coalition_info.  Sender_id = {str(sender_id)} with Coalition {str(msg_params.get(MyMessage.MSG_ARG_KEY_COALITION_ID))} and version {str(msg_params.get(MyMessage.MSG_ARG_KEY_COALITION_VERSION))}.")
            self.coalition_manager.handle_coalition_info_and_pigiback_update(msg_params, sender_id)
        except Exception:
               logging.exception("fallback add_result_for_flock failed")
               

    def run_sync(self):
        with raise_MPI_error():
            for epoch in range(self.epochs):
                if self.worker_index == 0:
                    rss, cuda = mem_mb()
                    logging.info(f"[rank {self.worker_index}] RSS={rss:.1f}MB CUDA={cuda:.1f}MB after dataloader ready")
                self.epoch = epoch
                n_bits = 0
                self.epoch_init()
                
                # if self.worker_index == 0 and epoch % 10 == 0:
                #     logging.info(f"[rank {self.worker_index}] Epoch {epoch} start, self.worker.min_sim = {self.worker.min_sim}, and self.worker.max_sim = {self.worker.max_sim}")
                for iteration in range(self.worker.num_iterations):
                    if self.round >= self.finish_round:
                        self.worker.is_last_step = True
                    self.iteration = iteration
                    # if (self.iteration % 50 == 0) and (self.worker_index == 0):
                    #     logging.info(cpu_mem_str(f"[rank {self.worker_index}] it={self.iteration}, q={self._rx_queue.qsize()}"))

                    # Train first, and then transfer the model
                    if self.args.Failure_chance is not None and np.random.rand(1) < self.args.Failure_chance:
                        logging.info("Communication Failure happens on worker: {}, Failure_chance: {}".format(
                            self.worker_index, self.args.Failure_chance))
                    else:
                        logging.debug("Start training on worker: {}, Epoch: {}, Iteration: {}".format(
                            self.worker_index, self.epoch, self.iteration))
                        time_before_train = time.time()
                        self.lr_schedule(self.epoch, self.iteration, self.global_round_idx,
                                         self.worker.num_iterations, self.args.warmup_epochs)
                        # update x_half to x_{t+1} by SGD
                        loss, output, target = self.worker.train_one_step(
                            self.epoch, self.iteration, self.train_tracker, self.metrics
                        )
                        time_after_train = time.time()
                        # if self.worker_index == 0 and self.iteration == 0:
                        #     logging.info("Worker %d, Epoch %d, Iteration %d, Train Time: %.2f" %
                        #                  (self.worker_index, self.epoch, self.iteration,
                        #                   time_after_train - time_before_train))

                    time_before_compress = time.time()
                    # Get model params
                    params, self.worker.shapes = get_data(
                        self.worker.param_groups, self.worker.param_names, is_get_grad=False
                    )
                    # if self.round >= 0:
                    flatten_params = TensorBuffer(params)
                    # ----- for real push-sum -----
                    # if self.worker_index == 0:
                    #     if self.worker.aggregate_cnt <10 or self.round < 20 or self.worker.aggregate_cnt % 50 == 1:
                    #         logging.info("Befor compression: Worker {}, Epoch {}, Iteration {}, with unpacked params {} after aggregation".format(
                    #             self.worker_index, self.epoch, self.iteration, params[0].flatten()[0:10]))
                    flatten_params.buffer.mul_(0.5)  # apply push-sum weight (halve local params)
                    # Update GPU and CPU param snapshots for cosine similarity
                    src = flatten_params.buffer
                    dst = self.worker.step_param_cpu
                    if self.worker._use_cuda and src.is_cuda:
                        # Store GPU param snapshot for cos sim (no copy, just reference)
                        # self.worker.step_param_gpu = src.detach()
                        new_snap = src.detach()  
                        old = self.worker.step_param_gpu
                        self.worker.step_param_gpu = new_snap
                        del old 
                    else:
                        if (dst is None) or (dst.numel() != src.numel()) or (dst.dtype != src.dtype):
                            self.worker.step_param_cpu = torch.empty_like(src, device='cpu', dtype=src.dtype)
                            # GPU -> CPU copy (non-blocking)
                            self.worker.step_param_cpu.copy_(src, non_blocking=True)

                    # compress
                    sync_buffer = {
                        "original_shapes": self.worker.shapes,
                        "flatten_params": flatten_params,
                    }
                    self.compressor.compress(sync_buffer)
                    self.selected_shapes = sync_buffer["selected_shapes"]

                    # Choose a neighbor to send (Hedonic gossip)
                    neighbor_idx = self.coalition_manager._choose_neighbor()
                    if self.compression in ["randomk", "topk"]:
                        n_bits += sync_buffer["n_bits"]
                        self.send_sparse_params_to_neighbors(
                            neighbor_idx, self.round,
                            sync_buffer["flatten_selected_values"].buffer.cpu(),
                            sync_buffer["flatten_selected_indices"].buffer.cpu(),
                            sync_buffer["selected_shapes"],
                            self.worker.get_dataset_len()
                        )
                    else:
                        raise NotImplementedError

                    time_after_compress = time.time()
                    # if self.worker_index == 0 and self.iteration == 0:
                    #     logging.info("Worker %d, Epoch %d, Iteration %d, Compress Time: %.2f" %
                    #                 (self.worker_index, self.epoch, self.iteration,
                    #                 time_after_compress - time_before_compress))

                    total_neighbor_cnt = 0
                    time_before = time.time()


                    has_payload = self.worker.has_payload_for_merge()
                    if not has_payload:
                        flatten_params.buffer.div_(0.5)
                        # if self.worker_index == 0:
                        #     if self.worker.aggregate_cnt <10 or self.round < 20 or self.worker.aggregate_cnt % 50 == 1:
                        #         logging.info("elif not has_payload: Worker {}, Epoch {}, Iteration {}, Aggregate {} neighbors, with unpacked params {} before aggregation".format(
                        #             self.worker_index, self.epoch, self.iteration, total_neighbor_cnt, params[0].flatten()[0:10]))
                    else:
                        if self.worker.is_last_step == False: 
                            total_neighbor_cnt += self.worker.number_of_neighbor_param_received
                            self.worker.aggregate_for_flock(self.compressor, sync_buffer["original_shapes"], sync_buffer)
                            # if self.worker_index == 0:
                            #     if self.worker.aggregate_cnt <10 or self.round < 20 or self.worker.aggregate_cnt % 50 == 1:
                            #         logging.info("if has_payload and self.worker.is_last_step == False: Worker {}, Epoch {}, Iteration {}, Aggregate {} neighbors, with unpacked params {} before aggregation".format(
                            #             self.worker_index, self.epoch, self.iteration, total_neighbor_cnt, params[0].flatten()[0:10]))
                        else:
                            while self.worker.has_payload_for_merge():
                                total_neighbor_cnt += self.worker.number_of_neighbor_param_received
                                self.worker.aggregate_for_flock(self.compressor, sync_buffer["original_shapes"], sync_buffer)
                                flatten_params.buffer.mul_(0.5)
                            # if self.worker_index == 0:
                            #     if self.worker.aggregate_cnt <10 or self.round < 20 or self.worker.aggregate_cnt % 50 == 1:
                            #         logging.info("elif has_payload and self.worker.is_last_step == True: Worker {}, Epoch {}, Iteration {}, Aggregate {} neighbors, with unpacked params {} before aggregation".format(
                            #             self.worker_index, self.epoch, self.iteration, total_neighbor_cnt, params[0].flatten()[0:10]))
                            flatten_params.buffer.div_(0.5)
                    time_after = time.time()
                    # if (self.worker_index == 0 and self.iteration == 0) or (self.worker_index == 0 and self.iteration == 10):
                    #     logging.info("Round %d, Iteration %d, while_cnt: %.2f, Time: %.2f" %
                    #                 (self.round, self.iteration, total_neighbor_cnt, time_after - time_before))
                    # Write back updated params to model
                    sync_buffer["flatten_params"].unpack(params)
                    # if self.worker_index == 0:
                    #     if self.worker.aggregate_cnt <10 or self.round < 20 or self.worker.aggregate_cnt % 50 == 1:
                    #         logging.info("After aggregation: Worker {}, Epoch {}, Iteration {}, Aggregate {} neighbors, with unpacked params {} after aggregation".format(
                    #             self.worker_index, self.epoch, self.iteration, total_neighbor_cnt, params[0].flatten()[0:10]))

                    # Cleanup large temporary buffers
                    try:
                        del sync_buffer["flatten_selected_values"]
                        del sync_buffer["flatten_selected_indices"]
                    except KeyError:
                        pass
                    # try:
                    #     del flatten_params
                    # except Exception:
                    #     pass

                    self.round += 1
                    # if self.round % 10 == 0:
                    #     logging.info(f"showing the coalition info of worker {self.worker_index} at round {self.round-1}: coalition set {self.coalition_manager._get_coalition_neighbor()}")


                    if (self.iteration % 50) == 0:
                        gc.collect()
                # End of iteration loop

                time_before_test = time.time()
                t = time.time()
                ts_unix = int(t) 
                msec   = int((t - ts_unix) * 1000)
                asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f",{msec:03d}"
                self.test_and_log(epoch,asctime)
                time_after_test = time.time()
                # if self.worker_index == 0:
                #     logging.info("Worker %d, Epoch %d, Iteration %d, Test Time: %.2f" %
                #                  (self.worker_index, self.epoch, self.iteration,
                #                   time_after_test - time_before_test))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # End of epoch loop
            logging.info("Worker {} finished all epochs, and received {} params from neighbors.".format(self.worker_index, self.msg_received))
            self.send_notify_to_coordinator(0)
            self._linger_after_training(2)


    def _linger_after_training(self, test_period_sec=2):
        """
        训练结束后继续监听并合并一段时间：
        - 若在 quiet_sec 内没有新的 payload 且队列为空 → 认为充分混合，退出
        - 最长等待 max_wait_sec（保险）
        - 期间每 test_period_sec 调一次 self.test_and_log(...)
        """

        # Get model params
        params, self.worker.shapes = get_data(
            self.worker.param_groups, self.worker.param_names, is_get_grad=False
        )

        while True:
            
            # if self.round >= 0:
            flatten_params = TensorBuffer(params)

            flatten_params.buffer.mul_(0.5)  # apply push-sum weight (halve local params)
            # Update GPU and CPU param snapshots for cosine similarity
            src = flatten_params.buffer
            dst = self.worker.step_param_cpu
            if self.worker._use_cuda and src.is_cuda:
                new_snap = src.detach()  
                old = self.worker.step_param_gpu
                self.worker.step_param_gpu = new_snap
                del old 
            else:
                if (dst is None) or (dst.numel() != src.numel()) or (dst.dtype != src.dtype):
                    self.worker.step_param_cpu = torch.empty_like(src, device='cpu', dtype=src.dtype)
                    # GPU -> CPU copy (non-blocking)
                    self.worker.step_param_cpu.copy_(src, non_blocking=True)

            # compress
            sync_buffer = {
                "original_shapes": self.worker.shapes,
                "flatten_params": flatten_params,
            }

            total_neighbor_cnt = 0

            has_payload = self.worker.has_payload_for_merge()
            if not has_payload:
                flatten_params.buffer.div_(0.5)
            else:
                total_neighbor_cnt += self.worker.number_of_neighbor_param_received
                self.worker.aggregate_for_flock(self.compressor, sync_buffer["original_shapes"], sync_buffer)
            sync_buffer["flatten_params"].unpack(params)
            try:
                del sync_buffer["flatten_selected_values"]
                del sync_buffer["flatten_selected_indices"]
            except KeyError:
                pass
            # try:
            #     del flatten_params
            # except Exception:
            #     pass

            if total_neighbor_cnt > 0:
                t = time.time()
                ts_unix = int(t) 
                msec   = int((t - ts_unix) * 1000)
                asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f",{msec:03d}"
                self.test_and_log(self.epochs-1,asctime)
                logging.info("Worker {} after finished all epochs, and received {} params from neighbors, {} this round.".format(self.worker_index, self.msg_received, total_neighbor_cnt))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(test_period_sec)



    def refresh_gossip_info(self):
        self.neighbors_info = self.topology_manager.topology
        self.gossip_info = self.topology_manager.topology[self.worker_index]

    def send_result_to_neighbors(self, receive_id, client_params1, local_sample_number):
        logging.debug("send_result_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PARAMS_1, client_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)

    def send_sparse_params_to_neighbors(self, receive_id, round, client_sparse_params1, client_sparse_index1, selected_shapes, local_sample_number):
        logging.debug("send_sparse_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_SPARSE_PARAMS_1, client_sparse_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_SPARSE_INDEX_1, client_sparse_index1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_ROUND, round)
        message.add_params(MyMessage.MSG_ARG_KEY_SELECTED_SHAPES, selected_shapes)
        self.coalition_manager.add_self_coalition_info_and_pigiback(message)
        self.send_message(message)
        # self.send_self_coalition_and_pigiback(receive_id)

    def send_self_coalition_and_pigiback(self,receive_id):
        # 2. send coalition and deltas
        message = Message(MyMessage.MSG_TYPE_COAL_INFO, self.get_sender_id(), receive_id)
        self.coalition_manager.add_self_coalition_info_and_pigiback(message)
        self.send_message(message)

    def send_active_coalition_join_notification(self,receive_id):
        message = Message(MyMessage.MSG_TYPE_COAL_INFO, self.get_sender_id(), receive_id)
        self.coalition_manager.add_self_coalition_info(message)
        self.send_message(message)



    def send_quant_params_to_neighbors(self, receive_id, client_quant_params1, local_sample_number):
        logging.debug("send_quant_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_QUANT_PARAMS_1, client_quant_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)

    def send_sign_params_to_neighbors(self, receive_id, client_sign_params1, local_sample_number):
        logging.debug("send_sign_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_SIGN_PARAMS_1, client_sign_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self.send_message(message)
    
    def send_notify_to_coordinator(self, receive_id=0):
        logging.debug("send_notify_to_coordinator. receive_id = %s, round: %s" %
                      (str(receive_id), str(self.round)))
        message = Message(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR, self.get_sender_id(), receive_id)
        self.send_message(message)

    def check_whether_all_clients_finish_receive(self):
        for rank, flag in self.flag_client_finish_dict.items():
            if not flag:
                return False
        return True

    def check_worker_finish_and_notify(self):
        if self.check_whether_all_clients_finish_receive():
            logging.debug(">>>>>>>>>>>>>>>COORDINATOR Receive all, ROUND %d finished!<<<<<<<<" %
                          (self.round))
            self.finish()

    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    def _update_neighbor_utility(self, nbr: int, cos_sim: float, staleness: int):
        # update_time = time.time()
        """更新 util_cache，然后自调 util_threshold，使 good_set 占比≈30%。"""
        
        bw   = self.bw_norm[nbr]

        phase1_round = self.epochs * self.worker.num_iterations * self.PHASE1
        if self.round < phase1_round:
            t    = self.round 
            lam_bw  = self.lambda_bw0
            lam_sim = 0
        else:
            t_adj    = self.round - phase1_round
            lam_bw  = max(self.lambda_bw0 * np.exp(- t_adj / self.time_const), 0.01 * self.lambda_bw0)  
            lam_sim = self.lambda_sim0 * (1 - np.exp(- t_adj / self.time_const))
            # lam_bw = 0.5
            # lam_sim = 0.5
        util    = lam_bw * bw + lam_sim * cos_sim - self.lambda_dist * (1-staleness)
        self.util_cache[nbr] = (1 - self.util_alpha) * self.util_cache[nbr] + self.util_alpha * util
        
        #     logging.info(f"[Round {self.round}] Worker {self.worker_index} update neighbor {nbr}: "
        #              f"util={self.util_cache[nbr]:.4f}, bw={bw:.3f}, sim={cos_sim:.3f}, stale={staleness}"
        #              f", lam_bw={lam_bw:.4f}, lam_sim={lam_sim:.4f}"
        #              f", lam_bw * bw={lam_bw * bw:.4f}, lam_sim * cos_sim={lam_sim * cos_sim:.4f},self.lambda_dist * (1-staleness)={self.lambda_dist * (1-staleness):.2f}")

        
        if self.util_cache[nbr] >= self.util_threshold:
            self.good_set.add(nbr); self.bad_set.discard(nbr)
        else:
            self.bad_set.add(nbr);  self.good_set.discard(nbr)

        
        outs = self.outs  
        if outs:
            good_ratio = len(self.good_set.intersection(outs)) / len(outs)
            if good_ratio < self.target_good_ratio - self.tolerance_ratio:
                self.util_threshold -= self.adjust_step  
            elif good_ratio > self.target_good_ratio + self.tolerance_ratio:
                self.util_threshold += self.adjust_step  

            
            for n in outs:
                if self.util_cache[n] >= self.util_threshold:
                    self.good_set.add(n); self.bad_set.discard(n)
                else:
                    self.bad_set.add(n);  self.good_set.discard(n)
        # update_time = time.time() - update_time
        # logging.info("update_neighbor_utility time %.4f" % update_time)


    def _choose_neighbor(self):  
        choose_time = time.time()
        """选择一个邻居进行发送。
        · 阶段①(前 15%)  : 纯均匀随机  —— 充分探索 / 快速扩散
        · 阶段②(15%~70%): 偏好 good_idx —— Soft‑max + ε/|bad|
        · 阶段③(>=70%)  : 只在 good_idx Soft‑max，ε→0
        同时始终保证  time_window  内每个 outs 至少被选一次。"""
        outs  = self.outs
        cur_r = self.round

        
        overdue = [n for n in outs if cur_r - self.last_chosen_round[n] >= self.time_window]
        if overdue:
            choice = np.random.choice(overdue)
            self.last_chosen_round[choice] = cur_r

            # if self.worker_index == 5:
            #     logging.debug(f"[Round {self.round}] Overdue....Worker 5 chose neighbor {choice} ")
            return int(choice)

        
        total_rounds = self.epochs * self.worker.num_iterations + 1e-8
        progress = cur_r / total_rounds

        PHASE1 = self.PHASE1  
        PHASE3 = self.PHASE3  

        
        if progress < PHASE1:
            # if self.worker_index == 5:
            #     logging.info(f"[Round {self.round}] Worker {self.worker_index} in Phase 1, self.util_cache: {self.util_cache[self.good_set]}")
            logits = self.bw_norm[outs]  
            probs  = np.exp(logits - logits.max())
            probs  = probs / probs.sum()
            choice = np.random.choice(outs, p=probs)
            self.last_chosen_round[choice] = cur_r
            return int(choice)
            # choice = np.random.choice(outs)
            # self.last_chosen_round[choice] = cur_r
            # # if self.worker_index == 5:
            # #     logging.debug(f"[Round {self.round}] Worker 5 chose neighbor {choice} ")
            # return int(choice)


        # choice = np.random.choice(outs)
        # self.last_chosen_round[choice] = cur_r
        # return int(choice)


        
        good_idx = [n for n in outs if n in self.good_set]
        bad_idx  = [n for n in outs if n in self.bad_set]

        
        if progress >= PHASE3 and good_idx:
            eps     = 0.05
            # logits = self.util_cache[good_idx] - np.max(self.util_cache[good_idx])
            # probs  = np.exp(logits); probs = probs / probs.sum()
            # choice = np.random.choice(good_idx, p=probs)
            # self.last_chosen_round[choice] = cur_r
            # # if self.worker_index == 5:
            # #     logging.info(f"[Round {self.round}] Worker 5 chose neighbor {choice} ")
            # return int(choice)

        # if self.round == self.epochs * self.worker.num_iterations * self.PHASE1:
        #     if self.worker_index == 5:
        #         logging.info(f"[Round {self.round}] Worker {self.worker_index} in Phase 2, self.util_cache: {self.util_cache[good_idx]}")
        
        eps     = self.util_eps  
        prob_g, prob_b = np.array([]), np.array([])

        if good_idx:
            logits = self.util_cache[good_idx]
            logits = logits - logits.max()
            w_g    = np.exp(logits)
            prob_g = (1 - eps) * w_g / w_g.sum()
        if bad_idx:
            prob_b = np.ones(len(bad_idx)) * (eps / len(bad_idx))

        all_probs = np.concatenate([prob_g, prob_b]); all_idxs = good_idx + bad_idx
        all_probs = all_probs / all_probs.sum()
        choice    = np.random.choice(all_idxs, p=all_probs)
        self.last_chosen_round[choice] = cur_r
        # if self.worker_index == 5:
        #     logging.info(f"[Round {self.round}] Worker 5 chose neighbor {choice} ")
        return int(choice)

    def my_get_out_neighbor_idx_list(self, worker_idx: int):
        """
        返回指定节点可发送的邻居列表。
        规则：同一行带宽 > 0 且非对角元素即为出边。
        若带宽矩阵已归一化，0 表示不可达。
        """
        assert hasattr(self, "bandwidth"), "请先生成并归一化 self.bandwidth"
        row = self.bandwidth[worker_idx]
        
        outs = [j for j, bw in enumerate(row) if (j != worker_idx) and (bw > 0)]
        return outs

    def _on_neighbor_eval(self, neighbor_idx: int, sender_round: int, cos_sim: float):
        """Callback for neighbor evaluation (staleness can be derived if needed)."""
        try:
            staleness = max(0, self.round - sender_round)
        except:
            staleness = 0
        self.coalition_manager._update_neighbor_utility(neighbor_idx, cos_sim, staleness, self.round)



    def _rx_loop(self):
        counter = 0
        while not self._rx_stop.is_set():
            try:
                msg_params = self._rx_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
                sender_round = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_ROUND)

                
                self._decode_tokens.acquire()

                def _decode_job(item, _sender_id, _sender_round):
                    """
                    在线程池里做 CPU 解压，构造一个“极轻”的 decoded 字典：
                    只携带 sender/round + 已解压的 q_values/q_indices。
                    注意：清理 item（释放张量引用）要在这里做，避免与主线程竞态。
                    """
                    self.msg_received += 1
                    
                    # if (self.msg_received <= 10)  and self.worker_index == 0:
                    #     logging.info(f"[rx] msg_received={self.msg_received}, rx-q={self._rx_queue.qsize()}, decoded-q={self._decoded_queue.qsize()}")  

                    enqueued = False
                    try:
                        
                        q_vals, q_idx = self.worker._rx_comp.decode_sparse_msg(item, self.worker._shapes)  

                        
                        if (q_vals is None) or (q_idx is None) or (q_vals.numel() == 0):
                            return

                        
                        q_vals = q_vals.contiguous()
                        q_idx  = q_idx.contiguous()

                        
                        decoded = {
                            MyMessage.MSG_ARG_KEY_SENDER: _sender_id,
                            MyMessage.MSG_ARG_KEY_LOCAL_ROUND: _sender_round,
                            "__decoded__": True,
                            "__q_values": q_vals,
                            "__q_indices": q_idx,
                        }

                        
                        self._decoded_queue.put(decoded)
                        enqueued = True
                        # if (self.msg_received <= 10)  and self.worker_index == 0:
                        #     logging.info(f"[rx] msg_received={self.msg_received}, rx-q={self._rx_queue.qsize()}, decoded-q={self._decoded_queue.qsize()}")  

                    except Exception:
                        logging.exception("decode_job failed")
                    finally:
                        
                        try:
                            if isinstance(item, dict):
                                item.clear()
                        except Exception:
                            pass
                        
                        if not enqueued:
                            try:
                                self._decode_tokens.release()
                            except Exception:
                                pass

                
                self._decode_pool.submit(_decode_job, msg_params, sender_id, sender_round)

                # add_time = 0.0
                # if self.worker_index == 0 and (counter % 500) == 0:
                #     logging.info("rx-io add_result_for_flock time %.4f with local counter %d", add_time, counter)
                counter += 1

                
                self.handle_msg_with_coalition_info(msg_params)

            except Exception:
                logging.exception("rx-io thread failed to schedule decode")
            finally:
                try:
                    self._rx_queue.task_done()
                except Exception:
                    pass


    def _agg_loop(self):
        """
        从已解压队列逐条取出消息 → 调用原 add_result_for_flock（免解压快路径） →
        释放令牌（允许下一条解压启动）。保证“已解压驻留 ≤ 3”且 push-sum 不丢包。
        """
        cnt = 0
        while not self._agg_stop.is_set():
            try:
                msg = self._decoded_queue.get(timeout=0.2)
                # if (self.msg_received <= 10)  and self.worker_index == 0:
                #             logging.info(f"[agg] msg_received={self.msg_received}, rx-q={self._rx_queue.qsize()}, decoded-q={self._decoded_queue.qsize()}")  
            except Empty:
                continue

            try:
                sender_id = msg.get(MyMessage.MSG_ARG_KEY_SENDER)
                sender_round = msg.get(MyMessage.MSG_ARG_KEY_LOCAL_ROUND)
                
                with get_lock(self.neighbor_transfer_lock):
                    t0 = time.time()
                    self.worker.add_result_for_flock(sender_id, sender_round, msg)
                    dt = time.time() - t0
                # if self.worker_index == 0 and (cnt % 500) == 0:
                #     logging.info("agg-io add_result_for_flock time %.4f cnt %d", dt, cnt)
                cnt += 1
            except Exception:
                logging.exception("agg-io thread failed in add_result_for_flock")
            finally:
                
                try:
                    self._decode_tokens.release()
                except Exception:
                    pass
                
                try:
                    if isinstance(msg, dict):
                        msg.pop("__q_values", None)
                        msg.pop("__q_indices", None)
                        msg.clear()
                except Exception:
                    pass
                try:
                    self._decoded_queue.task_done()
                except Exception:
                    pass
    
                
    @torch.no_grad()
    def snapshot_flatten_enqueue(self, flatten_buf, epoch:int):
        t = time.time()
        ts_unix = int(t) 
        msec   = int((t - ts_unix) * 1000)
        asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f",{msec:03d}"
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.worker.device)
            torch.cuda.synchronize(self.worker.device)  

        
        snap_cpu = flatten_buf.buffer.detach().cpu().clone()
        del flatten_buf
        gc.collect()

        
        meta = {
            "asctime": asctime,
            "epoch": epoch,
            "numel": int(snap_cpu.numel()),
            "dtype": str(snap_cpu.dtype),
            "shape": list(snap_cpu.shape),
            "worker": int(self.worker_index),
        }

        
        ts = self.snapshot_writer.enqueue(epoch, snap_cpu, meta)
        
        return f"./param_snapshots/{self.worker_index}/ep{epoch:04d}_{ts}"

    @torch.no_grad()
    def replay_and_test_all_epochs(self, params, pattern="ep*.pt"):
        """
        依次从磁盘恢复每个 epoch 的扁平参数快照 → unpack 到模型 → 直接调用 self.test_and_log(epoch)。
        默认读取 ./<worker_index>/epXXXX_*.pt。
        """
        import os, re, glob, torch

        
        snap_dir = f"./param_snapshots/{self.worker_index}"
        paths = glob.glob(os.path.join(snap_dir, pattern))

        
        def _parse_epoch(p):
            m = re.search(r"ep(\d+)_", os.path.basename(p))
            # logging.info(f"[replay] _parse_epoch {p} m={m}")
            return int(m.group(1)) if m else -1
        paths.sort(key=_parse_epoch)

        if not paths:
            print(f"[replay] no snapshots under: {snap_dir}")
            return

        

        
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(self.worker.device)

        
        with torch.no_grad():
            for p in paths:
                # logging.info(f"[replay] found {p} snapshots under: {snap_dir}")
                epoch = _parse_epoch(p)
                
                meta_path = os.path.splitext(p)[0] + ".json"
                snap_asctime = "N/A"
                
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        snap_asctime = meta.get("asctime", snap_asctime)
                    except Exception as e:
                        logging.warning(f"[replay] meta read failed: {meta_path}, err={e}")

                
                # sd_flat_cpu = torch.load(p, map_location="cpu")
                try:
                    sd_flat_cpu = torch.load(p, map_location="cpu", weights_only=True)
                    
                except TypeError:
                    sd_flat_cpu = torch.load(p, map_location="cpu")  

                
                
                
                buf = TensorBuffer(params)
                
                buf.buffer.copy_(sd_flat_cpu.to(buf.buffer.dtype))

                del sd_flat_cpu
                buf.unpack(params)

                
                if use_cuda:
                    for p in params:
                        if hasattr(p, "data") and hasattr(p.data, "to"):
                            p.data = p.data.to(self.worker.device)
                # if use_cuda:
                #     params.to(self.worker.device)

                
                self.test_and_log(epoch,snap_asctime)


 