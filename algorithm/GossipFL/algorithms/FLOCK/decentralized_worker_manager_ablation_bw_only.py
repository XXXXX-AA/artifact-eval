import numpy as np

from .decentralized_worker_manager_auto_backpressure import (
    DecentralizedWorkerManager as BaseDecentralizedWorkerManager,
)


class DecentralizedWorkerManager(BaseDecentralizedWorkerManager):
    """
    Ablation: bandwidth-only neighbor selection (no phase scheduling).
    """

    def _update_neighbor_utility(self, nbr: int, cos_sim: float, staleness: int):
        bw = self.bw_norm[nbr]
        util = bw
        self.util_cache[nbr] = (1 - self.util_alpha) * self.util_cache[nbr] + self.util_alpha * util

    # def _choose_neighbor(self):
    #     outs = self.outs
    #     cur_r = self.round

    #     overdue = [n for n in outs if cur_r - self.last_chosen_round[n] >= self.time_window]
    #     if overdue:
    #         choice = np.random.choice(overdue)
    #         self.last_chosen_round[choice] = cur_r
    #         return int(choice)

    #     logits = self.bw_norm[outs]
    #     probs = np.exp(logits - logits.max())
    #     probs = probs / probs.sum()
    #     choice = np.random.choice(outs, p=probs)
    #     self.last_chosen_round[choice] = cur_r
    #     return int(choice)

    def _choose_neighbor(self):
        outs = self.outs
        cur_r = self.round

        overdue = [n for n in outs if cur_r - self.last_chosen_round[n] >= self.time_window]
        if overdue:
            choice = np.random.choice(overdue)
            self.last_chosen_round[choice] = cur_r
            return int(choice)

        # NOTE: comment translated from Chinese
        choice = np.random.choice(outs)
        self.last_chosen_round[choice] = cur_r
        return int(choice)