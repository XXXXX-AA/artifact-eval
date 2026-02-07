import time
import logging
import threading
from collections import deque
from copy import deepcopy

import numpy as np
import torch

from fedml_core.distributed.communication.message import Message
from algorithms.baseDecent.decentralized_worker_manager import BaseDecentralizedWorkerManager
from .message_define import MyMessage
from .utils import transform_list_to_tensor, transform_tensor_to_list


def _flatten(state_dict):
    parts, shapes = [], {}
    for k, v in state_dict.items():
        t = v.detach().float().view(-1)
        shapes[k] = (v.shape, t.numel())
        parts.append(t)
    flat = torch.cat(parts) if parts else torch.tensor([])
    return flat, shapes


def _unflatten(flat, shapes, device):
    out, idx = {}, 0
    for k, (shape, numel) in shapes.items():
        if numel == 0:
            out[k] = torch.empty(shape, device=device)
        else:
            out[k] = flat[idx:idx + numel].view(shape).to(device)
        idx += numel
    return out


class DecentralizedWorkerManager(BaseDecentralizedWorkerManager):
    """
    AD-PSGD (aligned with the DPSGD structure):
      * use run_sync as the main training loop
      * each step only calls worker.train_one_step() (no data loading here)
      * tick-style async communication: _drain_rx() + _maybe_send()
      * bounded staleness filter + beta mixing (vectorized)
    """
    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics):
        super().__init__(args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics)

        # NOTE: comment translated from Chinese
        self.beta = float(getattr(args, "adpsgd_beta", 0.5))  # NOTE: comment translated from Chinese
        self.staleness_tau = int(getattr(args, "adpsgd_tau", 16))  # NOTE: comment translated from Chinese
        self.push_interval = float(getattr(args, "adpsgd_push_interval", 0.2))  # NOTE: comment translated from Chinese
        self.sample_neighbors = int(getattr(args, "adpsgd_sample_neighbors", 1))  # NOTE: comment translated from Chinese

        self.local_step = 0
        self._last_push = 0.0
        self.model_lock = threading.RLock()

        # NOTE: comment translated from Chinese
        self.rx_queue = deque(maxlen=4096)

    # NOTE: comment translated from Chinese
    def register_message_receive_handlers(self):
        # NOTE: comment translated from Chinese
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self._on_neighbor_model
        )

    def _on_neighbor_model(self, msg_params):
        try:
            sender = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
            step = int(msg_params.get(MyMessage.MSG_ARG_KEY_STEP, 0))
            payload = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
            self.rx_queue.append((sender, step, payload))
        except Exception as e:
            logging.exception(f"[AD-PSGD][RX-HANDLER] {e}")

    # NOTE: comment translated from Chinese
    def _drain_rx(self, max_ops=8):
        ops = 0
        while self.rx_queue and ops < max_ops:
            sender, step, payload = self.rx_queue.popleft()
            # NOTE: comment translated from Chinese
            if self.local_step - step > self.staleness_tau:
                ops += 1
                continue
            try:
                nbr_state = transform_list_to_tensor(payload)  # dict[str, Tensor]
                with self.model_lock:
                    cur_state = deepcopy(self.worker.get_model_params())
                    flat_cur, shapes = _flatten(cur_state)
                    flat_nbr, _ = _flatten(nbr_state)
                    if flat_cur.numel() == 0 or flat_nbr.numel() == 0:
                        ops += 1
                        continue
                    mixed = (1.0 - self.beta) * flat_cur + self.beta * flat_nbr
                    new_state = _unflatten(mixed, shapes, self.worker.device)
                    self.model_trainer.set_model_params(new_state)
            except Exception as e:
                logging.exception(f"[AD-PSGD][MIX] {e}")
            ops += 1

    # NOTE: comment translated from Chinese
    def _maybe_send(self):
        now = time.time()
        if now - self._last_push < self.push_interval:
            return
        outs = self.topology_manager.get_out_neighbor_idx_list(self.worker_index)
        if not outs:
            self._last_push = now
            return
        targets = np.random.choice(outs, size=min(self.sample_neighbors, len(outs)), replace=False)
        with self.model_lock:
            # NOTE: comment translated from Chinese
            state = deepcopy(self.worker.get_model_params())
            payload = transform_tensor_to_list(state)
            step = self.local_step
        for nid in targets:
            msg = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.worker_index, nid)
            msg.add_params(MyMessage.MSG_ARG_KEY_SENDER, self.worker_index)
            msg.add_params(MyMessage.MSG_ARG_KEY_RECEIVER, nid)
            msg.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, payload)
            msg.add_params(MyMessage.MSG_ARG_KEY_STEP, step)
            msg.add_params(MyMessage.MSG_ARG_KEY_TIMESTAMP, now)
            self.send_message(msg)
        self._last_push = now

    # NOTE: comment translated from Chinese
    def run_sync(self):
        """
        Aligned with DPSGD's run_sync:
          Per step:
            1) drain neighbors (_drain_rx)
            2) train one step (train_one_step)
            3) maybe send (_maybe_send)
            4) drain neighbors again (_drain_rx)
            5) optional periodic test/report
        """
        test_itv = int(getattr(self.args, "test_interval_steps", 200))
        max_steps = int(getattr(self.args, "max_local_steps", 10 ** 9))

        while self.local_step < max_steps and (not getattr(self, "_stop", False)):
            # NOTE: comment translated from Chinese
            self._drain_rx(max_ops=4)

            # NOTE: comment translated from Chinese
            try:
                with self.model_lock:
                    self.worker.train_one_step()
                self.local_step += 1
            except Exception as e:
                logging.exception(f"[AD-PSGD][train_one_step] {e}")
                break

            # NOTE: comment translated from Chinese
            self._maybe_send()

            # NOTE: comment translated from Chinese
            self._drain_rx(max_ops=4)

            # NOTE: comment translated from Chinese
            if test_itv > 0 and self.local_step % test_itv == 0:
                try:
                    # NOTE: comment translated from Chinese
                    self.report_metrics_if_available()
                except Exception:
                    pass

    # NOTE: comment translated from Chinese
    def report_metrics_if_available(self):
        if hasattr(self.worker, "test_on_ray"):
            try:
                train_metrics, test_metrics = self.worker.test_on_ray()
                msg = Message(MyMessage.MSG_TYPE_CLIENT_TO_COORDINATOR, self.worker_index, 0)
                msg.add_params("train_metrics", train_metrics)
                msg.add_params("test_metrics", test_metrics)
                self.send_message(msg)
            except Exception:
                pass

    def finish(self):
        self._stop = True
        super().finish()
