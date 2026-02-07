import logging
import numpy as np
from .utils import generate_bandwidth
from fedml_core.distributed.communication.message import Message

from utils.context import (
    raise_MPI_error,
    raise_error_without_process,
    get_lock,
)

from .message_define import MyMessage
import threading

from queue import Queue, Empty, Full
class CoalitionManager:
    def __init__(self, args, worker_index, size, bandwidth, bw_norm, outs, epochs, num_iterations):

        self.args = args
        self.worker_index = worker_index
        self.size = size
        
        self.round = 0
        self.epochs = epochs
        self.num_iterations = num_iterations

        # NOTE: comment translated from Chinese
        self.bandwidth = bandwidth  # NOTE: comment translated from Chinese
        bw_raw = self.bandwidth[self.worker_index]     # 1-D
        self.bw_norm = bw_norm   # 0-1
        logging.info(f"self.bw_norm : {self.bw_norm}")
        self.outs = outs
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


        # ====== Coalition (lazy gossip) state ======
        # NOTE: comment translated from Chinese
        self.coalition_index: int = int(self.worker_index)
        self.coalition_ver:   int = 0

        # NOTE: comment translated from Chinese
        # self._neighbor_coal_cache = {}   # {nid: (ci, ver, ts)}
        self._neighbor_coal_cache = {
        nid: (nid, 0, 0) for nid in range(size)
        }
        # NOTE: comment translated from Chinese
        # self._coal_lock_name = f"coalition-lock-{self.worker_index}"

        # self._coal_lock = get_lock(f"coalition-lock-{self.worker_index}")
        self._coal_lock = threading.Lock()

        # NOTE: comment translated from Chinese
        from collections import deque
        self._coal_delta_q = deque(maxlen=128)

        # NOTE: comment translated from Chinese
        self._last_broadcast_tick = 0
        self._broadcast_min_interval_tick = 3  # NOTE: comment translated from Chinese

        # NOTE: comment translated from Chinese
        self._target_density = 0.3  # NOTE: comment translated from Chinese
        self._tau0 = 0.02  # NOTE: comment translated from Chinese
        self._beta = 0.05  # NOTE: comment translated from Chinese

        self._active_coalition_join_cb = None
    
    def set_active_coalition_join_callback(self, aj):
        self._active_coalition_join_cb = aj
    

    # ---------- Coalition helpers ----------
    def _mono(self) -> int:
        """本地离散时钟：每调用一次当作 '1 tick'；只用于本地冷却与TTL。"""
        # NOTE: comment translated from Chinese
        return int(self.round)

    def _get_coalition_neighbor(self):
        same = 0
        coalition_neighbors = []
        outs = getattr(self, "outs", None) or []
        if not outs:
            return 0.0
        same = 0
        my_ci = self.coalition_index
        for n in outs:
            meta = self._neighbor_coal_cache.get(n)
            if meta and meta[0] == my_ci:
                same += 1
                coalition_neighbors.append(n)
        return same, coalition_neighbors
    
    def _local_coalition_density(self) -> float:
        """仅基于“已知邻居标签缓存”计算：与我同 ci 的邻居占比。"""
        outs = getattr(self, "outs", None) or []
        # if not outs:
        #     return 0.0
        # same = 0
        # my_ci = self.coalition_index
        # for n in outs:
        #     meta = self._neighbor_coal_cache.get(n)
        #     if meta and meta[0] == my_ci:
        #         same += 1
        same, _ = self._get_coalition_neighbor()
        return (same+1) / float(len(outs))

    def _tau_join(self, dens_i: float) -> float:
        """自适应换盟阈值：联盟越拥挤，越容易离开（τ 变小）。"""
        # τ = max(0, τ0 - β * max(0, dens_i - target))
        return max(0.0, self._tau0 - self._beta * max(0.0, dens_i - self._target_density))

    def _drain_coalition_deltas(self, k: int = 3):
        """
        每次发送前取出至多 k 条“未过期”的 (member, ci, ver) 做 piggyback。
        只做本地节流：基于离散 tick 控制出队频率；不过期直接带走。
        """
        out, tmp = [], []
        tick = self._mono()
        if tick - self._last_broadcast_tick < self._broadcast_min_interval_tick:
            return out
        self._last_broadcast_tick = tick

        while self._coal_delta_q and len(out) < k:
            member, ci, ver, ts = self._coal_delta_q.popleft()
            out.append((int(member), int(ci), int(ver)))
        # NOTE: comment translated from Chinese
        return out
    

    def add_self_coalition_info(self, message: Message):
        message.add_params(MyMessage.MSG_ARG_KEY_COALITION_ID, self.coalition_index)
        message.add_params(MyMessage.MSG_ARG_KEY_COALITION_VERSION, self.coalition_ver)

    def add_coalition_deltas(self, message: Message):
        """发送消息前调用，附加 piggyback 的 (member, ci, ver) 列表。"""
        deltas = self._drain_coalition_deltas(k=3)
        if deltas:
            message.add_params(MyMessage.MSG_ARG_KEY_DELTAS, deltas)


    def add_self_coalition_info_and_pigiback(self, message: Message):
        """发送消息前调用，附加自己的 (ci, ver) + piggyback 的 (member, ci, ver) 列表。"""
        self.add_self_coalition_info(message)
        self.add_coalition_deltas(message)

    def coalition_join_and_send_notification(self, inviter_id: int, inviter_ci: int, inviter_ver: int):
        """
        覆盖式换盟（不广播）：
        - ver = max(self.ver, inviter_ver) + 1
        - 回执 inviter 一个 ACK（轻量），并把 (self, ci, ver) 放进懒传播队列
        """
        with self._coal_lock:
            new_ci  = int(inviter_ci)
            new_ver = max(int(self.coalition_ver), int(inviter_ver)) + 1

            # NOTE: comment translated from Chinese
            self.coalition_index = new_ci
            self.coalition_ver   = new_ver
            now = self._mono()
            self._neighbor_coal_cache[self.worker_index] = (new_ci, new_ver, now)
            self._coal_delta_q.append((self.worker_index, new_ci, new_ver, now))
            
            # NOTE: comment translated from Chinese
            if callable(self._active_coalition_join_cb):
                try:
                    self._active_coalition_join_cb(int(inviter_id))
                except Exception:
                    logging.exception("eval callback failed in coalition_join_and_send_notification")

    def handle_coalition_join(self, msg_params, sender_id):
        try:
            # NOTE: comment translated from Chinese
            ci_ver = msg_params.get(MyMessage.MSG_TYPE_COAL_INFO) or {}
            ci = int(ci_ver.get(MyMessage.MSG_ARG_KEY_COALITION_ID, -1))
            ver = int(ci_ver.get(MyMessage.MSG_ARG_KEY_COALITION_VERSION, -1))
            if ci >= 0 and ver >= 0:
                with self._coal_lock:
                    # NOTE: comment translated from Chinese
                    old = self._neighbor_coal_cache.get(sender_id)
                    if (old is None) or (ver > old[1]):
                        now = self._mono()
                        self._neighbor_coal_cache[sender_id] = (ci, ver, now)
                        self._coal_delta_q.append((sender_id, ci, ver, now))
                        logging.info(f"worker {self.worker_index}, handle_coalition_join from worker {sender_id} of old coalition {old}.{old[1]} to {ci}.{ver}, and now the self._neighbor_coal_cache is {self._neighbor_coal_cache}")
        

        except Exception:
            logging.exception("handle_coalition_join handling failed")

    def handle_coalition_info_and_pigiback_update(self, msg_params, sender_id):
        # NOTE: comment translated from Chinese
        # NOTE: comment translated from Chinese
        # NOTE: comment translated from Chinese
        # NOTE: comment translated from Chinese
        # -----------------------------------------------
        try:
            # NOTE: comment translated from Chinese
            ci_ver = msg_params
            ci = int(ci_ver.get(MyMessage.MSG_ARG_KEY_COALITION_ID))
            ver = int(ci_ver.get(MyMessage.MSG_ARG_KEY_COALITION_VERSION))
            deltas = None
            params = ci_ver.get_params()  # NOTE: comment translated from Chinese
            if MyMessage.MSG_ARG_KEY_DELTAS in params:
                deltas = params[MyMessage.MSG_ARG_KEY_DELTAS]
            # if MyMessage.MSG_ARG_KEY_DELTAS in ci_ver:
            #     deltas = ci_ver.get(MyMessage.MSG_ARG_KEY_DELTAS)
            
            if ci >= 0 and ver >= 0:
                with self._coal_lock:
                    # NOTE: comment translated from Chinese
                    old = self._neighbor_coal_cache.get(sender_id)
                    if (old is not None) and (ver <= old[1]):
                        return
                    now = self._mono()
                    self._neighbor_coal_cache[sender_id] = (ci, ver, now)
                    self._coal_delta_q.append((sender_id, ci, ver, now))

                    
                    if deltas:
                        for tup in deltas:
                            try:
                                member, ci, ver = int(tup[0]), int(tup[1]), int(tup[2])
                            except Exception:
                                continue
                            with self._coal_lock:
                                old = self._neighbor_coal_cache.get(member)
                                if (old is not None) and (ver <= old[1]):
                                    continue
                                self._neighbor_coal_cache[member] = (ci, ver, now)
                                self._coal_delta_q.append((member, ci, ver, now))
                    # logging.info(f"worker {self.worker_index}, gets info from worker {sender_id}, and now the self._neighbor_coal_cache is {self._neighbor_coal_cache}")
        except Exception:
            logging.exception("coalition_info_and_pigiback_update handling failed")

    
    def switch_coalition(self, sender_id):
        # NOTE: comment translated from Chinese
        # NOTE: comment translated from Chinese
        sender_id_coalition_info = self._neighbor_coal_cache.get(sender_id)
        if sender_id_coalition_info:
            ci = sender_id_coalition_info[0]
            ver = sender_id_coalition_info[1]
            my_ci = self.coalition_index
            if ci != my_ci:
                # in_sender_coal = []
                try:
                    # in_sender_coal = [n for n in self.outs
                    #         if (self._neighbor_coal_cache.get(n) and
                    #             self._neighbor_coal_cache[n][0] == ci)]
                    # if in_sender_coal:
                    #     util_sender = float(np.mean([self.util_cache[n] for n in in_sender_coal]))
                    
                    util_sender = float(self.util_cache[sender_id])
                except Exception:
                    util_sender = 0.0
                outs = self.outs or []
                in_coal = [n for n in outs
                            if (self._neighbor_coal_cache.get(n) and
                                self._neighbor_coal_cache[n][0] == my_ci)]
                # logging.info(f"worker {self.worker_index} has in_coal {in_coal}, and outs {outs}")
                if in_coal:
                    u_curr = float(np.mean([self.util_cache[n] for n in in_coal]))
                else:
                    u_curr = 0.0
                gain = util_sender - u_curr
                dens_i = self._local_coalition_density()
                tau_join = self._tau_join(dens_i)
                # logging.info(f"worker {self.worker_index} has self._neighbor_coal_cache: {self._neighbor_coal_cache}")
                # logging.info(f"worker {self.worker_index} is considering switch_coalition, sender {sender_id} in coalition id {ci}has coalitions {in_sender_coal} and average utility: {util_sender}, and local average util: {u_curr}, local coalition density: {dens_i} and {tau_join}")
                if gain > tau_join and ci != my_ci:
                # if gain > 0 and ci != my_ci:
                    # logging.info(f"worker {self.worker_index} is switch_coalition ing from {my_ci} to {ci}")
                    # NOTE: comment translated from Chinese
                    self.coalition_join_and_send_notification(sender_id, ci, ver)

                    # NOTE: comment translated from Chinese
                    # deltas = msg_params.get("coal_deltas", None)
                    # if deltas:
                    #     self.handle_msg_coalition_join({"coal_deltas": deltas})

                    # NOTE: comment translated from Chinese
                    # if msg_params.get("msg_type") == "COALITION_JOIN_ACK":
                    #     self.handle_msg_coalition_join(msg_params)


    # ------------------------------------------------------------------
    # NOTE: comment translated from Chinese
    # ------------------------------------------------------------------
    def _update_neighbor_utility(self, nbr: int, cos_sim: float, staleness: int, round):
        self.round = round
        """更新 util_cache"""
        # NOTE: comment translated from Chinese
        bw   = self.bw_norm[nbr]

        phase1_round = self.epochs * self.num_iterations * self.PHASE1
        if self.round < phase1_round:
            lam_bw  = self.lambda_bw0
            lam_sim = 0
        else:
            t_adj    = self.round - phase1_round
            lam_bw  = max(self.lambda_bw0 * np.exp(- t_adj / self.time_const), 0.01 * self.lambda_bw0)  # NOTE: comment translated from Chinese
            lam_sim = self.lambda_sim0 * (1 - np.exp(- t_adj / self.time_const))
            # lam_bw = 0.5
            # lam_sim = 0.5
        util    = lam_bw * bw + lam_sim * cos_sim - self.lambda_dist * (1-staleness)
        self.util_cache[nbr] = (1 - self.util_alpha) * self.util_cache[nbr] + self.util_alpha * util

        self.switch_coalition(nbr)

        



    def _choose_neighbor(self):  # NOTE: comment translated from Chinese
        """选择一个邻居进行发送。
        · 阶段①(前 15%)  : 纯均匀随机  —— 充分探索 / 快速扩散
        · 阶段②(15%~70%): 偏好 good_idx —— Soft‑max + ε/|bad|
        · 阶段③(>=70%)  : 只在 good_idx Soft‑max，ε→0
        同时始终保证  time_window  内每个 outs 至少被选一次。"""
        outs  = self.outs
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
        total_rounds = self.epochs * self.num_iterations + 1e-8
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

        coalition_cnt, coalition_neighbor_idx = self._get_coalition_neighbor()
        non_coalition_cnt, non_coalition_neighbor_idx = len(self.outs)-coalition_cnt, [n for n in outs if n not in coalition_neighbor_idx]
        # logging.info(f"worker {self.worker_index} choosing neighbor from coalition set {coalition_cnt}, and out of coalition {non_coalition_cnt}")
        # good_idx = [n for n in outs if n in self.good_set]
        # bad_idx  = [n for n in outs if n in self.bad_set]

        # NOTE: comment translated from Chinese
        eps     = 0.05
        if progress >= PHASE3 and coalition_neighbor_idx:
            eps     = 0.05
            # logits = self.util_cache[good_idx] - np.max(self.util_cache[good_idx])
            # probs  = np.exp(logits); probs = probs / probs.sum()
            # choice = np.random.choice(good_idx, p=probs)
            # self.last_chosen_round[choice] = cur_r
            # # if self.worker_index == 5:
            # #     logging.info(f"[Round {self.round}] Worker 5 chose neighbor {choice} ")
            # return int(choice)

        # if self.round == self.epochs * self.num_iterations * self.PHASE1:
        #     if self.worker_index == 5:
        #         logging.info(f"[Round {self.round}] Worker {self.worker_index} in Phase 2, self.util_cache: {self.util_cache[good_idx]}")
        # NOTE: comment translated from Chinese
        eps     = self.util_eps  # NOTE: comment translated from Chinese
        prob_g, prob_b = np.array([]), np.array([])

        if coalition_neighbor_idx:
            logits = self.util_cache[coalition_neighbor_idx]
            logits = logits - logits.max()
            w_g    = np.exp(logits)
            prob_g = (1 - eps) * w_g / w_g.sum()
        if non_coalition_neighbor_idx:
            prob_b = np.ones(len(non_coalition_neighbor_idx)) * (eps / len(non_coalition_neighbor_idx))

        all_probs = np.concatenate([prob_g, prob_b]); all_idxs = coalition_neighbor_idx + non_coalition_neighbor_idx
        all_probs = all_probs / all_probs.sum()
        # logging.info(f"worker {self.worker_index} choosing neighbor from coalition set {coalition_cnt}, and out of coalition {non_coalition_cnt} with probabilities {all_probs}")
        
        choice    = np.random.choice(all_idxs, p=all_probs)
        self.last_chosen_round[choice] = cur_r
        # if self.worker_index == 5:
        #     logging.info(f"[Round {self.round}] Worker 5 chose neighbor {choice} ")
        return int(choice)
