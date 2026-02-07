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
        cpu = proc.cpu_percent(interval=None)  # NOTE: comment translated from Chinese
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
from .utils_randomk import generate_bandwidth

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

        # NOTE: comment translated from Chinese
        self.msg_received = 0
        self._rx_queue: Queue = Queue(maxsize=3) #2
        self._rx_stop = threading.Event()
        self._rx_thread = threading.Thread(
            name=f"rx-io-{rank}", target=self._rx_loop, daemon=True
        )
        # NOTE: comment translated from Chinese
        self.training_thread = threading.Thread(
            name="training", target=self.run_sync, daemon=True
        )

        # --- NEW: TX backpressure + ACK credits ---
        self.tx_peer_credit_init = int(getattr(args, "tx_peer_credit", 1)) # 2
        self.tx_global_tokens_init = int(getattr(args, "tx_global_tokens", 3)) # 6
        self.tx_queue_maxsize = int(getattr(args, "tx_queue_maxsize", 3)) # 6
        self._tx_queue: Queue = Queue(maxsize=self.tx_queue_maxsize)
        self._tx_stop = threading.Event()
        self._tx_thread = threading.Thread(
            name=f"tx-io-{rank}", target=self._tx_loop, daemon=True
        )
        self._tx_credit_lock = threading.Lock()
        self._tx_peer_tokens = [
            threading.Semaphore(self.tx_peer_credit_init) for _ in range(size)
        ]
        self._tx_peer_outstanding = [0 for _ in range(size)]
        self._tx_global_tokens = threading.Semaphore(self.tx_global_tokens_init)
        self._tx_global_outstanding = 0
        self._tx_send_pool = ThreadPoolExecutor(max_workers=max(1, self.tx_global_tokens_init))
        self.msg_sent = 0
        self.msg_acked = 0


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
        self._decode_tokens_total = int(getattr(args, "rx_decode_tokens", 3))
        self._decode_tokens = threading.BoundedSemaphore(value=self._decode_tokens_total)
        self._decode_tokens_lock = threading.Lock()
        self._decode_tokens_throttle = 0
        self._agg_stop = threading.Event()
        self._agg_thread = threading.Thread(
            name=f"agg-io-{rank}", target=self._agg_loop, daemon=True
        )

        # --- NEW: RSS-based adaptive backpressure ---
        self._bp_enabled = bool(getattr(args, "auto_backpressure", True))
        self._bp_interval_sec = float(getattr(args, "auto_bp_interval_sec", 0.5))
        self._bp_log_interval_sec = float(getattr(args, "auto_bp_log_interval_sec", 5.0))
        self._bp_last_log_t = 0.0
        self._bp_rx_min_tokens = int(getattr(args, "auto_bp_rx_min_tokens", 1))
        self._bp_tx_min_tokens = int(getattr(args, "auto_bp_tx_min_tokens", 1))
        self._bp_max_ack_delay_sec = float(getattr(args, "auto_bp_max_ack_delay_sec", 0.02))
        self._bp_ack_delay_sec = 0.0
        self._bp_stop = threading.Event()
        self._bp_thread = threading.Thread(
            name=f"bp-{rank}", target=self._auto_backpressure_loop, daemon=True
        )
        self._tx_global_total = self.tx_global_tokens_init
        self._tx_global_throttle = 0
        self._tx_global_lock = threading.Lock()
        self._rss_baseline_mb = mem_mb()[0]
        self._bp_rss_low_mb, self._bp_rss_high_mb = self._init_rss_thresholds(args)

        # whether all workers finished
        self.ifStop = False
        self.round = 0  # Current round index


        # NOTE: comment translated from Chinese
        self.bandwidth = generate_bandwidth(args)  # NOTE: comment translated from Chinese
        # if self.worker_index == 0:
        # logging.info(f"[rank {self.worker_index}] Bandwidth matrix:\n{self.bandwidth}")
        bw_raw = self.bandwidth[self.worker_index]     # 1-D
        bw_log_max = np.log1p(bw_raw).max() or 1.0  # NOTE: comment translated from Chinese
        self.bw_norm = np.log1p(bw_raw) / bw_log_max   # 0-1
        self.outs = self.my_get_out_neighbor_idx_list(self.worker_index)
        self.util_cache = np.zeros(size)  # NOTE: comment translated from Chinese
        self.util_alpha = 0.2  # NOTE: comment translated from Chinese
        self.util_eps = 0.3                            # ε‑greedy
        self.lambda_bw0 = 3.0
        self.lambda_sim0 = 0.8
        self.lambda_dist = 0.05 
        self.time_const = 1.0 * self.epochs
        # NOTE: comment translated from Chinese
        self.util_threshold = 0.0
        self.target_good_ratio = 0.3
        self.tolerance_ratio = 0.05
        self.adjust_step = 0.002
        self.time_window = int(np.ceil(3 * np.log(size)))
        self.last_chosen_round = np.zeros(size, dtype=int)

        # NOTE: comment translated from Chinese
        self.good_set = set()
        self.bad_set = set(range(size))

        self.PHASE1 = 0.04  # NOTE: comment translated from Chinese
        self.PHASE3 = 0.60  # NOTE: comment translated from Chinese
        if args.model == "resnet20":
            self.lambda_sim0 = 0.6#0.8
            self.lambda_dist = 0.1 #0.05
            self.PHASE1 = 0  # NOTE: comment translated from Chinese
            self.PHASE3 = 0.75  # NOTE: comment translated from Chinese
            self.time_const = 2.5 * self.epochs

        # compression part
        self.compression = args.compression
        assert self.compression in ["topk", "randomk", "quantize", "sign"]
        self.compressor = FLOCK_FLCompressor(comm_op=self.compression,
                                            compress_ratio=args.compress_ratio,  # args.compress_ratio
                                            quantize_level=args.quantize_level,
                                            is_biased=(args.is_biased == 1),)

        # NOTE: comment translated from Chinese
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
        self._tx_thread.start()
        if self._bp_enabled:
            self._bp_thread.start()

        if self.worker_index == 0:
            logging.debug("COORDINATOR notify clients to start!")
            self.coodinator_thread.start()
            self.notify_clients()
        super().run()

    def register_message_receive_handlers(self):
        # NOTE: comment translated from Chinese
        # self.register_message_receive_handler(MyMessage.MSG_TYPE_CLUSTER_INFO,
        #                                       self.handle_msg_cluster_info)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR,
                                              self.handle_msg_from_neighbor)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_ACK,
                                              self.handle_msg_ack_from_neighbor)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR,
                                              self.handle_msg_client_to_coordinator)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_COORDINATOR_TO_CLIENT,
                                              self.handle_msg_coordinator_to_client)

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
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        sender_round = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_ROUND)
        ack_ready = False
        try:
            #    self._rx_queue.put_nowait(msg_params)
            # self._rx_queue.put(msg_params, block=True, timeout=0.5)
            self._rx_queue.put(msg_params, block=True)
            ack_ready = True
        except Full:
            # NOTE: comment translated from Chinese
            try:
                logging.warning("rx queue full; process inline for sender=%s round=%s", sender_id, sender_round)
                with get_lock(self.neighbor_transfer_lock):
                    self.worker.add_result_for_flock(sender_id, sender_round, msg_params)
                ack_ready = True
            except Exception:
                logging.exception("fallback add_result_for_flock failed")
        if ack_ready and sender_id is not None:
            self._maybe_delay_ack()
            self._send_ack_to_neighbor(sender_id, sender_round)
        # NOTE: comment translated from Chinese
        return                                    

    def handle_msg_ack_from_neighbor(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        if sender_id is None:
            return
        self.msg_acked += 1
        self._release_tx_tokens(sender_id, reason="ack")

    def _send_ack_to_neighbor(self, receive_id, ack_round=None):
        message = Message(MyMessage.MSG_TYPE_ACK, self.get_sender_id(), receive_id)
        if ack_round is not None:
            message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_ROUND, ack_round)
        self.send_message(message)

    def _acquire_tx_tokens(self, peer_id):
        self._tx_peer_tokens[peer_id].acquire()
        self._tx_global_tokens.acquire()
        with self._tx_credit_lock:
            self._tx_peer_outstanding[peer_id] += 1
            self._tx_global_outstanding += 1

    def _release_tx_tokens(self, peer_id, reason="ack"):
        with self._tx_credit_lock:
            if peer_id is None:
                return
            if self._tx_peer_outstanding[peer_id] <= 0 or self._tx_global_outstanding <= 0:
                logging.warning(
                    "tx credit release ignored (peer=%s reason=%s peer_out=%s global_out=%s)",
                    peer_id,
                    reason,
                    self._tx_peer_outstanding[peer_id],
                    self._tx_global_outstanding,
                )
                return
            self._tx_peer_outstanding[peer_id] -= 1
            self._tx_global_outstanding -= 1
        try:
            self._tx_peer_tokens[peer_id].release()
        except Exception:
            pass
        try:
            self._tx_global_tokens.release()
        except Exception:
            pass

    def _init_rss_thresholds(self, args):
        rss_low_mb = float(getattr(args, "auto_bp_rss_low_mb", 0.0))
        rss_high_mb = float(getattr(args, "auto_bp_rss_high_mb", 0.0))
        if rss_low_mb > 0 and rss_high_mb > rss_low_mb:
            return rss_low_mb, rss_high_mb

        baseline = max(1.0, self._rss_baseline_mb)
        try:
            vm = psutil.virtual_memory()
            total_mb = vm.total / (1024 * 1024)
            avail_mb = vm.available / (1024 * 1024)
        except Exception:
            total_mb, avail_mb = 0.0, 0.0

        # default: low = baseline + 12% or +256MB; high = low + 20% or +512MB
        low = baseline + max(256.0, baseline * 0.12)
        high = low + max(512.0, baseline * 0.20)

        # If a global RSS budget is provided, align thresholds to it.
        budget = float(getattr(args, "auto_bp_rss_budget_mb", 0.0)) # 4000
        if budget > 0:
            low = min(low, max(1.0, budget * 0.75))
            high = min(high, max(low + 1.0, budget * 0.90))

        # Avoid exceeding a large fraction of system memory when available.
        if total_mb > 0:
            cap_high = total_mb * 0.35
            if avail_mb > 0:
                cap_high = min(cap_high, baseline + avail_mb * 0.60)
            if high > cap_high:
                high = max(low + 1.0, cap_high)
        return low, max(high, low + 1.0)

    def _apply_decode_throttle(self, target_tokens):
        target = max(self._bp_rx_min_tokens, min(self._decode_tokens_total, int(target_tokens)))
        with self._decode_tokens_lock:
            current = self._decode_tokens_total - self._decode_tokens_throttle
            delta = target - current
            if delta == 0:
                return current
            if delta > 0:
                # release held tokens to increase capacity
                release_n = min(delta, self._decode_tokens_throttle)
                for _ in range(release_n):
                    try:
                        self._decode_tokens.release()
                        self._decode_tokens_throttle -= 1
                    except Exception:
                        break
            else:
                # acquire tokens to reduce capacity (non-blocking)
                need = min(-delta, self._decode_tokens_total - self._bp_rx_min_tokens - self._decode_tokens_throttle)
                for _ in range(need):
                    got = self._decode_tokens.acquire(blocking=False)
                    if not got:
                        break
                    self._decode_tokens_throttle += 1
            return self._decode_tokens_total - self._decode_tokens_throttle

    def _apply_tx_throttle(self, target_tokens):
        target = max(self._bp_tx_min_tokens, min(self._tx_global_total, int(target_tokens)))
        with self._tx_global_lock:
            current = self._tx_global_total - self._tx_global_throttle
            delta = target - current
            if delta == 0:
                return current
            if delta > 0:
                release_n = min(delta, self._tx_global_throttle)
                for _ in range(release_n):
                    try:
                        self._tx_global_tokens.release()
                        self._tx_global_throttle -= 1
                    except Exception:
                        break
            else:
                need = min(-delta, self._tx_global_total - self._bp_tx_min_tokens - self._tx_global_throttle)
                for _ in range(need):
                    got = self._tx_global_tokens.acquire(blocking=False)
                    if not got:
                        break
                    self._tx_global_throttle += 1
            return self._tx_global_total - self._tx_global_throttle

    def _auto_backpressure_loop(self):
        ema = 0.0
        alpha = float(getattr(self.args, "auto_bp_ema_alpha", 0.2))
        while not self._bp_stop.is_set():
            try:
                rss_mb, _ = mem_mb()
                if self._bp_rss_high_mb <= self._bp_rss_low_mb:
                    pressure = 0.0
                else:
                    pressure = (rss_mb - self._bp_rss_low_mb) / (self._bp_rss_high_mb - self._bp_rss_low_mb)
                if pressure < 0:
                    pressure = 0.0
                elif pressure > 1:
                    pressure = 1.0
                ema = (1 - alpha) * ema + alpha * pressure

                rx_target = int(round(self._bp_rx_min_tokens + (1.0 - ema) * (self._decode_tokens_total - self._bp_rx_min_tokens)))
                tx_target = int(round(self._bp_tx_min_tokens + (1.0 - ema) * (self._tx_global_total - self._bp_tx_min_tokens)))

                rx_eff = self._apply_decode_throttle(rx_target)
                tx_eff = self._apply_tx_throttle(tx_target)
                self._bp_ack_delay_sec = max(0.0, min(self._bp_max_ack_delay_sec, ema * self._bp_max_ack_delay_sec))

                now = time.time()
                if now - self._bp_last_log_t >= self._bp_log_interval_sec:
                    logging.info(
                        "[rank %s] auto-bp rss=%.1fMB (low=%.1f high=%.1f) p=%.2f ema=%.2f rx=%s/%s tx=%s/%s ack=%.3fs",
                        self.worker_index,
                        rss_mb,
                        self._bp_rss_low_mb,
                        self._bp_rss_high_mb,
                        pressure,
                        ema,
                        rx_eff,
                        self._decode_tokens_total,
                        tx_eff,
                        self._tx_global_total,
                        self._bp_ack_delay_sec,
                    )
                    self._bp_last_log_t = now
            except Exception:
                logging.exception("auto backpressure loop failed")
            time.sleep(self._bp_interval_sec)

    def _maybe_delay_ack(self):
        if not self._bp_enabled:
            return
        delay = self._bp_ack_delay_sec
        if delay > 0:
            time.sleep(delay)

    def _enqueue_tx_message(self, receive_id, message):
        self._tx_queue.put(
            {"neighbor_idx": receive_id, "message": message},
            block=True,
        )

    def run_sync(self):
        with raise_MPI_error():
            for epoch in range(self.epochs):
                if self.worker_index == 0:
                    rss, cuda = mem_mb()
                    logging.info(f"[rank {self.worker_index}] RSS={rss:.1f}MB CUDA={cuda:.1f}MB after dataloader ready")
                self.epoch = epoch
                n_bits = 0
                self.epoch_init()
                if self.worker_index == 0 and epoch % 10 == 0:
                    logging.info(f"[rank {self.worker_index}] Epoch {epoch} start, self.worker.min_sim = {self.worker.min_sim}, and self.worker.max_sim = {self.worker.max_sim}")
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
                        new_snap = src.detach()  # NOTE: comment translated from Chinese
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
                    neighbor_idx = self._choose_neighbor()
                    if neighbor_idx is not None:
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

                    if (self.iteration % 50) == 0:
                        gc.collect()
                # End of iteration loop

                time_before_test = time.time()
                # record_path = self.snapshot_flatten_enqueue(flatten_params,epoch)
                # logging.info("test metrics record into %s" % record_path)
                # if epoch % 5 == 0 or epoch == self.epochs - 1:
                    # record_path = self.snapshot_flatten_enqueue(flatten_params,epoch)
                #     logging.info("test metrics record into %s" % record_path)
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
            # self.snapshot_writer.flush() 
            # self.replay_and_test_all_epochs(params)
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
                new_snap = src.detach()  # NOTE: comment translated from Chinese
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
        self._enqueue_tx_message(receive_id, message)

    def send_sparse_params_to_neighbors(self, receive_id, round, client_sparse_params1, client_sparse_index1, selected_shapes, local_sample_number):
        logging.debug("send_sparse_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_SPARSE_PARAMS_1, client_sparse_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_SPARSE_INDEX_1, client_sparse_index1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_ROUND, round)
        message.add_params(MyMessage.MSG_ARG_KEY_SELECTED_SHAPES, selected_shapes)
        self._enqueue_tx_message(receive_id, message)

    def send_quant_params_to_neighbors(self, receive_id, client_quant_params1, local_sample_number):
        logging.debug("send_quant_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_QUANT_PARAMS_1, client_quant_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self._enqueue_tx_message(receive_id, message)

    def send_sign_params_to_neighbors(self, receive_id, client_sign_params1, local_sample_number):
        logging.debug("send_sign_params_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_SIGN_PARAMS_1, client_sign_params1)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_number)
        self._enqueue_tx_message(receive_id, message)
    
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
    # NOTE: comment translated from Chinese
    # ------------------------------------------------------------------
    def _update_neighbor_utility(self, nbr: int, cos_sim: float, staleness: int):
        # update_time = time.time()
        """更新 util_cache，然后自调 util_threshold，使 good_set 占比≈30%。"""
        # NOTE: comment translated from Chinese
        bw   = self.bw_norm[nbr]

        phase1_round = self.epochs * self.worker.num_iterations * self.PHASE1
        if self.round < phase1_round:
            t    = self.round 
            lam_bw  = self.lambda_bw0
            lam_sim = 0
        else:
            t_adj    = self.round - phase1_round
            lam_bw  = max(self.lambda_bw0 * np.exp(- t_adj / self.time_const), 0.01 * self.lambda_bw0)  # NOTE: comment translated from Chinese
            lam_sim = self.lambda_sim0 * (1 - np.exp(- t_adj / self.time_const))
            # lam_bw = 0.5
            # lam_sim = 0.5
        # util    = lam_bw * bw + lam_sim * cos_sim - self.lambda_dist * (1-staleness)
        util    = lam_bw * bw + lam_sim * cos_sim - self.lambda_dist * staleness
        self.util_cache[nbr] = (1 - self.util_alpha) * self.util_cache[nbr] + self.util_alpha * util
        # NOTE: comment translated from Chinese
        #     logging.info(f"[Round {self.round}] Worker {self.worker_index} update neighbor {nbr}: "
        #              f"util={self.util_cache[nbr]:.4f}, bw={bw:.3f}, sim={cos_sim:.3f}, stale={staleness}"
        #              f", lam_bw={lam_bw:.4f}, lam_sim={lam_sim:.4f}"
        #              f", lam_bw * bw={lam_bw * bw:.4f}, lam_sim * cos_sim={lam_sim * cos_sim:.4f},self.lambda_dist * (1-staleness)={self.lambda_dist * (1-staleness):.2f}")

        # NOTE: comment translated from Chinese
        if self.util_cache[nbr] >= self.util_threshold:
            self.good_set.add(nbr); self.bad_set.discard(nbr)
        else:
            self.bad_set.add(nbr);  self.good_set.discard(nbr)

        # NOTE: comment translated from Chinese
        outs = self.outs  # NOTE: comment translated from Chinese
        if outs:
            good_ratio = len(self.good_set.intersection(outs)) / len(outs)
            if good_ratio < self.target_good_ratio - self.tolerance_ratio:
                self.util_threshold -= self.adjust_step  # NOTE: comment translated from Chinese
            elif good_ratio > self.target_good_ratio + self.tolerance_ratio:
                self.util_threshold += self.adjust_step  # NOTE: comment translated from Chinese

            # NOTE: comment translated from Chinese
            for n in outs:
                if self.util_cache[n] >= self.util_threshold:
                    self.good_set.add(n); self.bad_set.discard(n)
                else:
                    self.bad_set.add(n);  self.good_set.discard(n)
        # update_time = time.time() - update_time
        # logging.info("update_neighbor_utility time %.4f" % update_time)


    def _choose_neighbor(self):  # NOTE: comment translated from Chinese
        choose_time = time.time()
        """选择一个邻居进行发送。
        · 阶段①(前 15%)  : 纯均匀随机  —— 充分探索 / 快速扩散
        · 阶段②(15%~70%): 偏好 good_idx —— Soft‑max + ε/|bad|
        · 阶段③(>=70%)  : 只在 good_idx Soft‑max，ε→0
        同时始终保证  time_window  内每个 outs 至少被选一次。"""
        # NOTE: comment translated from Chinese
        outs = [n for n in self.outs if self.bandwidth[self.worker_index][n] > 0]
        if not outs:
            logging.warning(
                f"[Round {self.round}] Worker {self.worker_index} has no bw>0 neighbors; skip sending."
            )
            return None
        cur_r = self.round

        # NOTE: comment translated from Chinese
        overdue = [n for n in outs if cur_r - self.last_chosen_round[n] >= self.time_window]
        if overdue:
            choice = np.random.choice(overdue)
            self.last_chosen_round[choice] = cur_r

            # if self.worker_index == 5:
            #     logging.debug(f"[Round {self.round}] Overdue....Worker 5 chose neighbor {choice} ")
            return int(choice)

        # NOTE: comment translated from Chinese
        total_rounds = self.epochs * self.worker.num_iterations + 1e-8
        progress = cur_r / total_rounds

        PHASE1 = self.PHASE1  # NOTE: comment translated from Chinese
        PHASE3 = self.PHASE3  # NOTE: comment translated from Chinese

        # NOTE: comment translated from Chinese
        if progress < PHASE1:
            # if self.worker_index == 5:
            #     logging.info(f"[Round {self.round}] Worker {self.worker_index} in Phase 1, self.util_cache: {self.util_cache[self.good_set]}")
            logits = self.bw_norm[outs]  # NOTE: comment translated from Chinese
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


        # NOTE: comment translated from Chinese
        good_idx = [n for n in outs if n in self.good_set]
        bad_idx  = [n for n in outs if n in self.bad_set]
        eps     = 0.05
        # NOTE: comment translated from Chinese
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
        # NOTE: comment translated from Chinese
        else:
            eps     = self.util_eps  # NOTE: comment translated from Chinese
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
        # NOTE: comment translated from Chinese
        outs = [j for j, bw in enumerate(row) if (j != worker_idx) and (bw > 0)]
        return outs

    def _on_neighbor_eval(self, neighbor_idx: int, sender_round: int, cos_sim: float):
        """Callback for neighbor evaluation (staleness can be derived if needed)."""
        try:
            staleness = max(0, self.round - sender_round)
            STALE_CAP = 200
            stal = min(staleness, STALE_CAP)              # 0..20
            stal_norm = stal / STALE_CAP                  # 0..1
        except:
            staleness = 0
        self._update_neighbor_utility(neighbor_idx, cos_sim, stal_norm)



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

                # NOTE: comment translated from Chinese
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
                        # NOTE: comment translated from Chinese
                        q_vals, q_idx = self.worker._rx_comp.decode_sparse_msg(item, self.worker._shapes)  

                        # NOTE: comment translated from Chinese
                        if (q_vals is None) or (q_idx is None) or (q_vals.numel() == 0):
                            return

                        # NOTE: comment translated from Chinese
                        q_vals = q_vals.contiguous()
                        q_idx  = q_idx.contiguous()

                        # NOTE: comment translated from Chinese
                        decoded = {
                            MyMessage.MSG_ARG_KEY_SENDER: _sender_id,
                            MyMessage.MSG_ARG_KEY_LOCAL_ROUND: _sender_round,
                            "__decoded__": True,
                            "__q_values": q_vals,
                            "__q_indices": q_idx,
                        }

                        # NOTE: comment translated from Chinese
                        self._decoded_queue.put(decoded)
                        enqueued = True
                        # if (self.msg_received <= 10)  and self.worker_index == 0:
                        #     logging.info(f"[rx] msg_received={self.msg_received}, rx-q={self._rx_queue.qsize()}, decoded-q={self._decoded_queue.qsize()}")  

                    except Exception:
                        logging.exception("decode_job failed")
                    finally:
                        # NOTE: comment translated from Chinese
                        try:
                            if isinstance(item, dict):
                                item.clear()
                        except Exception:
                            pass
                        # NOTE: comment translated from Chinese
                        if not enqueued:
                            try:
                                self._decode_tokens.release()
                            except Exception:
                                pass

                # NOTE: comment translated from Chinese
                self._decode_pool.submit(_decode_job, msg_params, sender_id, sender_round)

                add_time = 0.0
                # if self.worker_index == 0 and (counter % 500) == 0:
                #     logging.info("rx-io add_result_for_flock time %.4f with local counter %d", add_time, counter)
                counter += 1

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
                # NOTE: comment translated from Chinese
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
                # NOTE: comment translated from Chinese
                try:
                    self._decode_tokens.release()
                except Exception:
                    pass
                # NOTE: comment translated from Chinese
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

    def _tx_loop(self):
        """
        TX backpressure:
        - 从 _tx_queue 取待发送消息
        - 获取 per-peer credit + global tokens
        - 异步 send_message()
        - ACK 到达后释放 tokens
        """
        while not self._tx_stop.is_set():
            try:
                tx_item = self._tx_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                neighbor_idx = tx_item.get("neighbor_idx")
                message = tx_item.get("message")
                if neighbor_idx is None or message is None:
                    continue
                if not isinstance(neighbor_idx, int) or neighbor_idx < 0 or neighbor_idx >= self.size:
                    continue

                self._acquire_tx_tokens(neighbor_idx)

                def _send_job(msg, nbr_idx):
                    try:
                        self.msg_sent += 1
                        self.send_message(msg)
                    except Exception:
                        logging.exception("send_job failed for neighbor %s", nbr_idx)
                        self._release_tx_tokens(nbr_idx, reason="send_fail")

                try:
                    self._tx_send_pool.submit(_send_job, message, neighbor_idx)
                except Exception:
                    self._release_tx_tokens(neighbor_idx, reason="schedule_fail")
                    logging.exception("tx-io thread failed to schedule send")

            finally:
                try:
                    self._tx_queue.task_done()
                except Exception:
                    pass
    
                
    @torch.no_grad()
    def snapshot_flatten_enqueue(self, flatten_buf, epoch:int):
        t = time.time()
        ts_unix = int(t) 
        msec   = int((t - ts_unix) * 1000)
        asctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f",{msec:03d}"
        # NOTE: comment translated from Chinese
        if torch.cuda.is_available():
            torch.cuda.set_device(self.worker.device)
            torch.cuda.synchronize(self.worker.device)  # NOTE: comment translated from Chinese

        # NOTE: comment translated from Chinese
        snap_cpu = flatten_buf.buffer.detach().cpu().clone()
        del flatten_buf
        gc.collect()

        # NOTE: comment translated from Chinese
        meta = {
            "asctime": asctime,
            "epoch": epoch,
            "numel": int(snap_cpu.numel()),
            "dtype": str(snap_cpu.dtype),
            "shape": list(snap_cpu.shape),
            "worker": int(self.worker_index),
        }

        # NOTE: comment translated from Chinese
        ts = self.snapshot_writer.enqueue(epoch, snap_cpu, meta)
        # NOTE: comment translated from Chinese
        return f"./param_snapshots/{self.worker_index}/ep{epoch:04d}_{ts}"

    @torch.no_grad()
    def replay_and_test_all_epochs(self, params, pattern="ep*.pt"):
        """
        依次从磁盘恢复每个 epoch 的扁平参数快照 → unpack 到模型 → 直接调用 self.test_and_log(epoch)。
        默认读取 ./<worker_index>/epXXXX_*.pt。
        """
        import os, re, glob, torch

        # NOTE: comment translated from Chinese
        snap_dir = f"./param_snapshots/{self.worker_index}"
        paths = glob.glob(os.path.join(snap_dir, pattern))

        # NOTE: comment translated from Chinese
        def _parse_epoch(p):
            m = re.search(r"ep(\d+)_", os.path.basename(p))
            # logging.info(f"[replay] _parse_epoch {p} m={m}")
            return int(m.group(1)) if m else -1
        paths.sort(key=_parse_epoch)

        if not paths:
            print(f"[replay] no snapshots under: {snap_dir}")
            return

        

        # NOTE: comment translated from Chinese
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(self.worker.device)

        # NOTE: comment translated from Chinese
        with torch.no_grad():
            for p in paths:
                # logging.info(f"[replay] found {p} snapshots under: {snap_dir}")
                epoch = _parse_epoch(p)
                
                meta_path = os.path.splitext(p)[0] + ".json"
                snap_asctime = "N/A"
                # NOTE: comment translated from Chinese
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        snap_asctime = meta.get("asctime", snap_asctime)
                    except Exception as e:
                        logging.warning(f"[replay] meta read failed: {meta_path}, err={e}")

                # NOTE: comment translated from Chinese
                # sd_flat_cpu = torch.load(p, map_location="cpu")
                try:
                    sd_flat_cpu = torch.load(p, map_location="cpu", weights_only=True)
                    
                except TypeError:
                    sd_flat_cpu = torch.load(p, map_location="cpu")  # NOTE: comment translated from Chinese

                # NOTE: comment translated from Chinese
                # NOTE: comment translated from Chinese
                
                buf = TensorBuffer(params)
                
                buf.buffer.copy_(sd_flat_cpu.to(buf.buffer.dtype))

                del sd_flat_cpu
                buf.unpack(params)

                # NOTE: comment translated from Chinese
                if use_cuda:
                    for p in params:
                        if hasattr(p, "data") and hasattr(p.data, "to"):
                            p.data = p.data.to(self.worker.device)
                # if use_cuda:
                #     params.to(self.worker.device)

                # NOTE: comment translated from Chinese
                self.test_and_log(epoch,snap_asctime)


 
