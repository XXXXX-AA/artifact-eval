# FLOCK Emulation (Anonymized)

## Prerequisites
- Linux (tested on Ubuntu 22.04)
- Docker Engine + Docker CLI
- Containernet + Mininet + Open vSwitch
- NVIDIA Container Toolkit (if using GPU)
- Python 3.8+ with docker SDK / requests / urllib3 compatible versions
- sudo privileges (Containernet requires root)

## Repository Layout
- `launch_scripts/`: emulation/launch scripts and topology tools
- `algorithm/`: algorithm and training code
  - `algorithm/GossipFL/`: GossipFL + FLOCK implementation
  - `algorithm/docker/`: Dockerfile for building `gossipfl:latest`

## Preparation
1. Build the Docker image (requires a `myfedml:mpi` base image, or edit the Dockerfile base image)
   ```bash
   cd algorithm/docker
   docker build -t gossipfl:latest .
   ```

2. Prepare datasets (optional)
   - Default location: `algorithm/GossipFL/data/`
   - To use another location, set `GOSSIPFL_DATA_DIR`

## Run Emulation
From the repository root:
```bash
cd launch_scripts
sudo /path/to/python3 launch_containernet_mpi_leaf_spine_fixed_v6.py \
  > ../logs/run-$(date +%Y%m%d_%H%M%S).log 2>&1 & disown
```

Logs are written under `logs/` (auto-created if missing).

## Optional Environment Variables
- `FLOCK_ROOT`: repository root (defaults to `launch_scripts/..`)
- `GOSSIPFL_DIR`: algorithm code directory (defaults to `algorithm/GossipFL`)
- `GOSSIPFL_DATA_DIR`: dataset directory (defaults to `algorithm/GossipFL/data`)
- `FLOCK_LOGS_DIR`: logs output directory (defaults to `logs/`)
- `FLOCK_SSHKEY_DIR`: mount an existing SSH key directory if needed

## Main Tunables
`algorithm/GossipFL/experiments/mpi_based/launch_mpi_based_docker.sh` reads:
- `algorithm/GossipFL/experiments/configs_system/gpu3_64_docker.conf`
- `algorithm/GossipFL/experiments/configs_model/resnet20.conf`
- `algorithm/GossipFL/experiments/configs_algorithm/CHOCO_SGD.conf`
- `algorithm/GossipFL/experiments/main_args.conf`

Edit those files to adjust worker count, model, algorithm, and training parameters.

## Notes
- Logs and datasets are not included in this anonymized package.
- Add more configs under `configs_*` if you need other algorithms/models.
