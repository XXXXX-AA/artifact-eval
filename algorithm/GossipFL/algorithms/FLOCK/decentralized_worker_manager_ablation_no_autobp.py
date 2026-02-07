import threading
from queue import Queue

from .decentralized_worker_manager_auto_backpressure import (
    DecentralizedWorkerManager as BaseDecentralizedWorkerManager,
)


class DecentralizedWorkerManager(BaseDecentralizedWorkerManager):
    """
    Ablation: disable auto backpressure and remove RX/TX throttling.
    """

    def __init__(self, args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics):
        super().__init__(args, comm, rank, size, worker, topology_manager, model_trainer, timer, metrics)

        # Disable auto backpressure loop.
        self._bp_enabled = False
        self._bp_ack_delay_sec = 0.0

        # Remove queue limits (unbounded).
        self._rx_queue = Queue(maxsize=0)
        self._decoded_queue = Queue(maxsize=0)
        self._tx_queue = Queue(maxsize=0)
        self.tx_queue_maxsize = 0

        # Make decode tokens effectively unbounded (no RX throttling).
        unbounded = int(getattr(args, "ablation_unbounded_tokens", 1000000))
        self._decode_tokens_total = unbounded
        self._decode_tokens = threading.Semaphore(value=unbounded)

    def _acquire_tx_tokens(self, peer_id):
        return

    def _release_tx_tokens(self, peer_id, reason="ack"):
        return

    def _maybe_delay_ack(self):
        return
