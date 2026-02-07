#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOCK_ROOT="$(realpath "${SCRIPT_DIR}/..")"
GOSSIPFL_DIR="${GOSSIPFL_DIR:-${FLOCK_ROOT}/algorithm/GossipFL}"
sudo docker system prune -f
sudo pkill -f launch_containernet_mpi.py || true
sudo mn -c
sudo docker ps -a --format '{{.Names}}' | grep '^mn\.' | xargs -r sudo docker rm -f


# 2) Remove leftover mn.worker* containers (if any)
docker ps -a -q -f name='^mn\.worker' | xargs -r docker rm -f

# 3) (Optional) Disconnect leftover endpoints from the bridge
for n in $(docker ps -a --format '{{.Names}}' | grep '^mn\.worker'); do
  docker network disconnect -f bridge "$n" || true
done

# 4) (Optional) Restart the Docker daemon to refresh network state
sudo systemctl restart docker

sudo rm -rf "${GOSSIPFL_DIR}/spill_queue"
sudo rm -rf "${GOSSIPFL_DIR}/data_preprocessing/cifar10/partitions/cifar10_part.json"
sudo rm -rf "${GOSSIPFL_DIR}/data_preprocessing/MNIST/partitions/mnist_part.json"
sudo rm -rf "${GOSSIPFL_DIR}/data_preprocessing/cifar100/partitions/cifar100_part.json"
sudo rm -rf "${GOSSIPFL_DIR}/algorithms/SAPS_FL/generate_bandwidth"
sudo rm -rf "${GOSSIPFL_DIR}/algorithms/CHOCO_SGD/choco-sgd-generate_bandwidth"
sudo rm -rf "${GOSSIPFL_DIR}/algorithms/DPSGD/dpsgd-generate_bandwidth"
# sudo rm -rf "${GOSSIPFL_DIR}/spill-rank*"
