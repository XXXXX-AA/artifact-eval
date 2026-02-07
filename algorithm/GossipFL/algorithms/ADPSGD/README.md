
# AD-PSGD (Asynchronous Decentralized SGD)

This module adapts your existing DPSGD scaffolding to an **asynchronous** variant in the style of AD-PSGD:

- No global barriers; local SGD runs continuously.
- Background **TX** thread periodically pushes the current model to sampled out-neighbors.
- Background **RX** thread **mixes** incoming neighbor models immediately using a constant mixing weight `beta`.
- A **staleness filter** drops too-old neighbor models if `local_step - sender_step > tau`.

## Key Args (with defaults)
- `--adpsgd_beta` (float, default 0.5): mixing weight between local and neighbor model (0.0 → keep local, 1.0 → overwrite by neighbor).
- `--adpsgd_tau` (int, default 16): bounded-staleness threshold in local-steps.
- `--adpsgd_push_interval` (seconds, default 0.2): interval between model pushes.
- `--adpsgd_sample_neighbors` (int, default 1): number of neighbors to push per interval.
- `--graph_degree` (int, default 2): out-degree for symmetric topology generation.

## How to Use
Replace the DPSGD API call with:
```python
from algorithms.ADPSGD.ADPSGD_API import FedML_ADPSGD
FedML_ADPSGD(process_id, worker_number, device, comm,
             model, train_data_num, train_data_global, test_data_global,
             train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args)
```
Your training/optimizer/loss configs from `MyModelTrainer.py` remain unchanged.

> Note: This is a **vanilla reference** AD-PSGD variant (constant mixing). If you want **degree-based** mixing or a proper **doubly-stochastic** W, replace the simple `beta` mixing in `decentralized_worker_manager.py::_rx_loop` with per-neighbor weights.
