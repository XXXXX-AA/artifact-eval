import logging
import time
from copy import deepcopy
import torch
from torch import nn
import threading
import traceback
from mpi4py import MPI

from utils.timer import Timer
from utils.data_utils import get_data, apply_gradient
from utils.tensor_buffer import TensorBuffer
from algorithms.baseDecent.decentralized_worker_FLOCK import BaseDecentralizedWorker


class DecentralizedWorker(BaseDecentralizedWorker):
    def __init__(self, worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number,
                 device, model, args, model_trainer, timer, metrics):

        self.min_sim, self.max_sim = self.get_sim_min_max(args)
        self.worker_index = worker_index
        self.topology_manager = topology_manager
        # self.refresh_gossip_info()
        self.is_last_step = False

        super().__init__(worker_index, topology_manager, train_data_global, test_data_global, train_data_num,
                         train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_number,
                         device, model, args, model_trainer, timer, metrics)

        # ---- model param meta ----
        self.param_groups = self.model_trainer.param_groups
        self.param_names = self.model_trainer.param_names
        self._shapes, self._total_len = self.init_flatten_params_shapes()
        self.compress_ratio = args.compress_ratio

        # ---- global switches / counters ----
        self._use_cuda = (torch.cuda.is_available() and ("cuda" in str(device)))
        self.neighbor_weight = 0.5
        self.number_of_neighbor_param_received = 0   # count for this round
        self.agg_number_of_neighbor_param_received = 0 # count for this aggregation
        self.total_neighbor_weight = 0.0
        self.aggregate_cnt = 0

        # snapshot for computing cosine similarity (optional)
        self.step_param_gpu = None
        self.step_param_cpu = None

        # injected later by manager
        self._rx_comp = None
        self._eval_cb = None

        # =========================
        #   CPU-only accum buffers
        # =========================
        if not self._use_cuda:

            
            self._rx_lock = threading.Lock()
            cpu_buf = torch.zeros(self._total_len, dtype=torch.float32, device="cpu")
            self._rx_active = cpu_buf  
            self._rx_standby = torch.zeros_like(cpu_buf)

            w_buf = torch.zeros(self._total_len, dtype=torch.float32, device="cpu")
            self.neighbor_total_weight_active = w_buf  
            self.neighbor_total_weight_standby = torch.zeros_like(w_buf)
        else:
            
            self._rx_lock = None
            self._rx_active = None
            self._rx_standby = None
            self.neighbor_total_weight_active = None
            self.neighbor_total_weight_standby = None


        # =========================
        #   GPU-only accum buffers
        # =========================
        if self._use_cuda:
            
            # === Reusable pinned CPU staging buffers (fixed, grow-once) ===
            self._pin_idx = None   # CPU pinned int64 buffer for indices
            self._pin_val = None   # CPU pinned float32 buffer for values

            
            
            try:
                get_range = getattr(torch.cuda, "get_stream_priority_range", None)
                torch.cuda.set_device(self.device)  
                if callable(get_range):
                    low, high = get_range()  
                    accum_pri = low  
                    self._accum_stream = torch.cuda.Stream(priority=accum_pri)
                    # logging.info(f"Worker {self.worker_index} stream device: {self._accum_stream.device}, 111priority range: ({low}, {high}), accum_pri: {accum_pri}")
                else:
                    
                    try:
                        self._accum_stream = torch.cuda.Stream(priority=0)  
                        # logging.info(f"Worker {self.worker_index} stream device: {self._accum_stream.device}, 222priority range: unknown, accum_pri: 0")
                    except TypeError:
                        
                        self._accum_stream = torch.cuda.Stream()
                        # logging.info(f"Worker {self.worker_index} stream device: {self._accum_stream.device}, 333priority range: unknown, accum_pri: default")
            except RuntimeError:
                
                self._accum_stream = torch.cuda.Stream()
            
            # logging.info(f"Worker {self.worker_index} stream device: {self._accum_stream.device}")

            # low, high = torch.cuda.get_stream_priority_range()
            # self._accum_stream = torch.cuda.Stream(priority=low)
            self._accum_lock = threading.Lock()
            self._accum_evt = torch.cuda.Event(blocking=False)
            self._evt_recorded = False  

            
            self._gpu_buf_hot = None  
            self._gpu_w_hot = None  

            self._gpu_buf_ready = None  
            self._gpu_w_ready = None
            
            self._ones_cache = None  # GPU only
        else:
            self._accum_stream = None
            self._accum_lock = None
            self._gpu_buf_hot = None
            self._gpu_w_hot = None
            self._gpu_buf_ready = None
            self._gpu_w_ready = None

        # stats
        self.stat_rx_msgs = 0
        self.stat_sim_computed = 0
        self.stat_batch_swaps = 0
        self.stat_batch_merge_h2d_time = 0.0

        
        self._state_lock = threading.Lock()
        self._payload_ready = False  
        self._agg_in_progress = False  


    def get_sim_min_max(self,args):
        # return 0.9, 1.0
        if args.dataset == "mnist":
            self.min_sim = 0.95
            self.max_sim = 1
        elif args.dataset == "cifar10" and args.model == "resnet56":
            self.min_sim = 0.9
            self.max_sim = 0.999
        elif args.dataset == "cifar10" and args.model == "resnet32":
            self.min_sim = 0.9
            self.max_sim = 0.999
        elif args.dataset == "cifar10" and args.model == "resnet20":
            self.min_sim = 0.9
            self.max_sim = 0.999
            # self.min_sim = 0.984
            # self.max_sim = 0.987
        elif args.dataset == "cifar10" and args.model == "cifar10flnet":
            self.min_sim = 0.9
            self.max_sim = 0.999
            # self.min_sim = 0.984
            # self.max_sim = 1
            # self.min_sim = 0.984
            # self.max_sim = 0.987
        elif args.dataset == "cifar100" and args.model == "efficientnet":
            self.min_sim = 0.9
            self.max_sim = 0.999
            # self.min_sim = 0.9859877824783325
            # self.max_sim = 0.9999988675117493
        return self.min_sim, self.max_sim

    # =========================
    #  Public setter from manager
    # =========================
    def set_rx_compressor(self, compressor):
        self._rx_comp = compressor

    def set_eval_callback(self, cb):
        self._eval_cb = cb

    def _ensure_pinned(self, key: str, numel: int, dtype):
        assert key in ('idx', 'val')
        buf = self._pin_idx if key == 'idx' else self._pin_val
        need_new = (buf is None) or (not buf.is_pinned()) or (buf.dtype != dtype) or (buf.numel() < numel)
        if need_new:
            new_buf = torch.empty(numel, dtype=dtype, pin_memory=True)
            if key == 'idx':
                self._pin_idx = new_buf
            else:
                self._pin_val = new_buf
            buf = new_buf
        return buf

    def _stage_to_pinned(self, cpu_tensor, key: str, dtype):
        assert cpu_tensor.device.type == 'cpu'
        t = cpu_tensor
        if not t.is_contiguous():
            t = t.contiguous()
        if t.dtype != dtype:
            t = t.to(dtype=dtype, copy=False)
        numel = t.numel()
        buf = self._ensure_pinned(key, numel, dtype)
        view = buf[:numel]
        view.copy_(t, non_blocking=True)
        return view

    # =========================
    #     Receive & Accumulate
    # =========================
    @torch.no_grad()
    def add_result_for_flock(self, worker_index, sender_round, updated_information):
        """
        解包邻居稀疏增量并累加到“当轮累加缓冲”。
        - GPU 训练：只使用 GPU 双缓冲（hot），绝不触碰 CPU 累加缓冲。
        - CPU 训练：只使用 CPU 双缓冲（_rx_active），绝不触碰 GPU 累加缓冲。
        """
        if self._rx_comp is None:
            raise RuntimeError("RX compressor not set. Call set_rx_compressor() before training.")

        try:
            # 1) decode on CPU
            uncompress_start = time.time()
            q_indices = None
            q_values = None
            # 1) decode on CPU
            if isinstance(updated_information, dict) and updated_information.get("__decoded__", False):
                q_values = updated_information.get("__q_values")
                q_indices = updated_information.get("__q_indices")
            else:
                q_values, q_indices = self._rx_comp.decode_sparse_msg(updated_information, self._shapes)
            uncompress_end = time.time()
            uncompress_time = uncompress_end - uncompress_start
            # if self.worker_index == 0 and (self.stat_rx_msgs % 500) == 0:
            #     logging.info("add_result_for_flock uncompress time %.4f with local round %d" % (uncompress_time, self.stat_rx_msgs))
            if (q_values is None) or (q_indices is None) or (q_values.numel() == 0):
                return
            add_start = time.time()
            # idx_cpu = q_indices.to(dtype=torch.long)
            idx_cpu = q_indices.to(dtype=torch.long, copy=False)
            val_cpu = q_values.to(dtype=torch.float32, copy=False)
            
            idx_pin = self._stage_to_pinned(idx_cpu, 'idx', torch.long)
            val_pin = self._stage_to_pinned(val_cpu, 'val', torch.float32)


            # ------------------- GPU path -------------------
            if self._use_cuda and (self._total_len > 0):
                torch.cuda.set_device(self.device)

                
                if not hasattr(self, "_h2d_stream"):
                    self._h2d_stream = torch.cuda.Stream(device=self.device)
                if not hasattr(self, "_h2d_evt"):
                    self._h2d_evt = torch.cuda.Event(blocking=False)

                with torch.cuda.stream(self._h2d_stream):
                    idx_gpu = idx_pin.to(self.device, non_blocking=True)
                    val_gpu = val_pin.to(self.device, non_blocking=True)
                    self._h2d_evt.record(self._h2d_stream)  
                    h2d_end = time.time()
                    h2d_end1 = h2d_end - add_start
                    # if self.worker_index == 0 and (self.stat_rx_msgs % 500) == 0:
                    #     logging.info("add_result_for_flock D2H time %.4f with local round %d" % (h2d_end1, self.stat_rx_msgs))

                
                self._accum_stream.wait_event(self._h2d_evt)
                
                cos_sim = 0.0
                with self._accum_lock:

                    if (self._gpu_buf_hot is None) or (self._gpu_buf_hot.numel() != self._total_len):
                        self._gpu_buf_hot = torch.zeros(self._total_len, dtype=torch.float32, device=self.device)
                        self._gpu_w_hot = torch.zeros_like(self._gpu_buf_hot)
                        self._ones_cache = torch.ones(len(idx_cpu), dtype=torch.float32, device=self.device)
                        
                        self._gpu_buf_ready = None
                        self._gpu_w_ready = None

                    
                    with torch.cuda.stream(self._accum_stream):
                        # add_start = time.time()
                        # idx_gpu = idx_pin.to(device=self.device, non_blocking=True)
                        # val_gpu = val_pin.to(device=self.device, dtype=torch.float32, non_blocking=True)
                        # h2d_end = time.time()
                        # h2d_end1 = h2d_end - add_start
                        # if self.worker_index == 0 and (self.stat_rx_msgs % 50) == 0:
                        #     logging.info("add_result_for_flock D2H time %.4f with local round %d" % (h2d_end1, self.stat_rx_msgs))
                        if (self.step_param_gpu is not None) and (self.step_param_gpu.numel() == self._total_len):
                            EPS = 1e-8
                            a = self.step_param_gpu[idx_gpu]
                            b = val_gpu
                            denom = a.norm() * b.norm() + EPS
                            if float(denom) > 0.0:
                                cs = torch.dot(a, b) / denom
                                # if cs < self.min_sim:
                                #     self.min_sim = cs
                                # if cs > self.max_sim:
                                #     self.max_sim = cs
                                cs = torch.clamp(cs, min=self.min_sim, max=self.max_sim)
                                cos_sim = float((cs - self.min_sim) / (self.max_sim - self.min_sim + EPS))
                        self.stat_sim_computed += 1 
                        sim_end = time.time()
                        sim_end1 = sim_end - h2d_end
                        # if self.worker_index == 0 and (self.stat_rx_msgs % 500) ==   0:
                        #     logging.info("add_result_for_flock sim time %.4f with local round %d" % (sim_end1, self.stat_rx_msgs))

                        self._gpu_buf_hot.index_add_(0, idx_gpu, val_gpu)
                        self._gpu_w_hot.index_add_(0, idx_gpu, self._ones_cache, alpha=float(self.neighbor_weight))
                        add_end = time.time()
                        add_end1 = add_end - sim_end
                        # if self.worker_index == 0 and (self.stat_rx_msgs % 500) == 0:
                        #     logging.info("add_result_for_flock index_add time %.4f with local round %d" % (add_end1, self.stat_rx_msgs))

                        
                        self._accum_evt.record(self._accum_stream)
                        self._evt_recorded = True
                        record_end = time.time()
                        record_end = record_end - add_end
                        # if self.worker_index == 0 and (self.stat_rx_msgs % 500) == 0:
                        #     logging.info("add_result_for_flock record time %.4f with local round %d" % (record_end, self.stat_rx_msgs))


                    
                    self.number_of_neighbor_param_received += 1
                    self.stat_rx_msgs += 1

                    
                    # if not hasattr(self, "_accum_done_evt"):
                    #     self._accum_done_evt = torch.cuda.Event(blocking=False)
                    # self._accum_done_evt.record(self._accum_stream)

                    
                with self._state_lock:
                    self._payload_ready = True


            # ------------------- CPU path -------------------
            else:
                
                cos_sim = 0.0
                if (self.step_param_cpu is not None) and (self.step_param_cpu.numel() == self._total_len):
                    local_slice = self.step_param_cpu[idx_cpu]
                    neigh_slice = q_values.to(local_slice.dtype)
                    EPS = 1e-8
                    denom = local_slice.norm() * neigh_slice.norm() + EPS
                    if denom > 0.0:
                        cs = torch.dot(local_slice, neigh_slice) / denom
                        cs = torch.clamp(cs, min=0.9, max=1.0)
                        cos_sim = (cs - 0.9) / (1.0 - 0.9 + EPS)
                        # cos_sim = float((cs - 0.9) / (1.0 - 0.9 + EPS))
                self.stat_sim_computed += 1

                
                with self._rx_lock:
                    if self.number_of_neighbor_param_received == 0:
                        self._rx_active.zero_()
                        self.neighbor_total_weight_active.zero_()

                    self._rx_active.index_add_(0, idx_cpu, q_values)
                    self.neighbor_total_weight_active[idx_cpu] += self.neighbor_weight
                    self.number_of_neighbor_param_received += 1
                    self.stat_rx_msgs += 1

                
                with self._state_lock:
                    self._payload_ready = True


            
            update_utility_start = time.time()
            if callable(self._eval_cb):
                try:
                    self._eval_cb(int(worker_index), int(sender_round), float(cos_sim))
                except Exception:
                    logging.exception("eval callback failed in add_result_for_flock (CPU)")
            update_utility_end = time.time()
            update_utility_time = update_utility_end - update_utility_start
            # if self.worker_index == 0 and (self.stat_rx_msgs % 500) == 0:
            #     logging.info("add_result_for_flock update_utility_time %.4f with local round %d" % (update_utility_time, self.stat_rx_msgs))
            # del q_values, idx_cpu, idx_gpu, val_gpu
            for _x in ("q_values","idx_cpu","idx_gpu","val_gpu"):
                _t = locals().get(_x, None)
                if _t is not None:
                    del _t


        except Exception:
            logging.exception("add_result_for_flock failed")

    # =========================
    #          Aggregate
    # =========================
    @torch.no_grad()
    def aggregate_for_flock(self, compressor, selected_shapes, sync_buffer):
        """
        将“当轮累加缓冲”聚合到本地参数：
          - GPU 训练：仅使用 GPU ready（由 hot 原子 flip 得到），按索引更新；未触达索引 /0.5 还原。
          - CPU 训练：仅使用 CPU ready（由 active/standby 交换得到），按索引更新；未触达索引 /0.5 还原。
        """
        self.aggregate_cnt += 1

        
        flatten_buf = sync_buffer["flatten_params"].buffer
        if flatten_buf.numel() != self._total_len:
            logging.warning("aggregate_for_flock: flatten size mismatch.")
            return
        
        with self._state_lock:
            if self._agg_in_progress:
                logging.warning("aggregate_for_flock: aggregation already in progress, skip.")
                return
            self._agg_in_progress = True
            payload_now = self._payload_ready
            # logging.info("Worker {} start aggregate_for_flock called, payload_now: {}, neighbor_msgs: {}, agg_cnt: {}.".format(
            #     self.worker_index, payload_now, self.number_of_neighbor_param_received, self.aggregate_cnt
            # ))
        
        
        if not payload_now:
            logging.info("aggregate_for_flock with payload_not ready")
            # sync_buffer["flatten_params"].buffer.div_(float(self.neighbor_weight))
            with self._state_lock:
                self._agg_in_progress = False
            return
        
        # ------------------- GPU path -------------------
        if self._use_cuda and flatten_buf.is_cuda:
            torch.cuda.set_device(self.device)
            # if self._accum_stream is not None:
            #     self._accum_stream.synchronize()

            
            with self._accum_lock:
                had_event = self._evt_recorded  
                if had_event:
                    # torch.cuda.current_stream().wait_event(self._accum_evt)
                    
                    with torch.cuda.device(self.device):
                        torch.cuda.current_stream().wait_event(self._accum_evt)
                        self._evt_recorded = False  

                if (self._gpu_buf_hot is None) or (self._gpu_w_hot is None):
                    
                    self._gpu_buf_ready = None
                    self._gpu_w_ready = None
                else:
                    if (self._gpu_buf_ready is None) or (self._gpu_buf_ready.numel() != self._total_len):
                        self._gpu_buf_ready = torch.zeros_like(self._gpu_buf_hot)
                        self._gpu_w_ready = torch.zeros_like(self._gpu_w_hot)
                        
                    self._gpu_buf_ready, self._gpu_buf_hot = self._gpu_buf_hot, self._gpu_buf_ready
                    self._gpu_w_ready, self._gpu_w_hot = self._gpu_w_hot, self._gpu_w_ready
                    self._gpu_buf_hot.zero_()
                    self._gpu_w_hot.zero_()

                
                
                self.agg_number_of_neighbor_param_received = self.number_of_neighbor_param_received
                self.number_of_neighbor_param_received = 0
                self.total_neighbor_weight = 0.0

            
            
            if (self._gpu_buf_ready is not None) and (self._gpu_w_ready is not None):
                # flatten_buf.add_(self._gpu_buf_ready)
                try:
                    flatten_buf.add_(self._gpu_buf_ready)
                    # flatten_buf.div_((self.agg_number_of_neighbor_param_received+1)*0.5)
                    # logging.info("aggregate_for_flock with self._gpu_buf_ready in Worker {}, agg_number_of_neighbor_param_received: {}.".format(
                    #     self.worker_index, self.agg_number_of_neighbor_param_received))
                    self._gpu_w_ready.add_(0.5).clamp_(min=1e-8).reciprocal_()
                    flatten_buf.mul_(self._gpu_w_ready)

                finally:
                    
                    self._gpu_buf_ready.zero_()
                    self._gpu_w_ready.zero_()
                    with self._state_lock:
                        
                        self._payload_ready = False
                        self._agg_in_progress = False

            else:
                
                try:
                    logging.info("aggregate_for_flock with self._gpu_buf_ready is None in Worker {}.".format(self.worker_index))
                    flatten_buf.div_(float(self.neighbor_weight))
                finally:
                    with self._state_lock:
                        
                        self._payload_ready = False
                        self._agg_in_progress = False

        # ------------------- CPU path -------------------
        else:
            
            if self.number_of_neighbor_param_received == 0:
                flatten_buf.div_(float(self.neighbor_weight))
                return

            
            with self._rx_lock:
                ready = self._rx_active
                ready_w = self.neighbor_total_weight_active
                
                self._rx_active, self._rx_standby = self._rx_standby, self._rx_active
                self.neighbor_total_weight_active, self.neighbor_total_weight_standby = \
                    self.neighbor_total_weight_standby, self.neighbor_total_weight_active
                
                self.number_of_neighbor_param_received = 0
                self.total_neighbor_weight = 0.0
            try:
                
                mask = ready_w > 0
                if mask.any():
                    numer = flatten_buf[mask] + ready[mask]                   # 0.5*local + sumΔ
                    denom = ready_w[mask] + float(self.neighbor_weight)       # sum(0.5) + 0.5
                    flatten_buf[mask] = numer / denom
                
                inv = (~mask)
                if inv.any():
                    flatten_buf[inv] = flatten_buf[inv] / float(self.neighbor_weight)
                # flatten_buf.add_(ready)
                # ready_w.add_(0.5).clamp_(min=1e-8).reciprocal_()
                # flatten_buf.mul_(ready_w)
                    
            finally:
                
                ready.zero_()
                ready_w.zero_()
                with self._state_lock:
                    
                    self._payload_ready = False
                    self._agg_in_progress = False
        

    # =========================
    #        Utilities
    # =========================
    def init_flatten_params_shapes(self):
        params, _shapes = get_data(self.param_groups, self.param_names, is_get_grad=False)
        if (params is None) or (len(params) == 0):
            logging.info("init_flatten_params_shapes: param list empty; delay shapes init.")
            return [], 0
        _total_len = int(sum(int(p.numel()) for p in params))
        return _shapes, _total_len

    # def refresh_gossip_info(self):
    #     self.neighbors_info = self.topology_manager.topology
    #     self.gossip_info = self.topology_manager.topology[self.worker_index]
    #     self.in_neighbor_idx_list = self.topology_manager.get_in_neighbor_idx_list(self.worker_index)


    def has_payload_for_merge(self) -> bool:
        with self._state_lock:
            return bool(self._payload_ready) and (not self._agg_in_progress)
