#!/bin/bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
# Initialize pyenv
if [ -d "$PYENV_ROOT" ]; then
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
fi
set -o noglob # Disable globbing to prevent wildcard expansion, solved the problem of checkpoint_epoch_list changing from [10,20,30] to 1

# Check if pyenv is available
echo "PYENV_ROOT: $PYENV_ROOT"
echo "PATH: $PATH"
which pyenv
######################################  mpi_based launch shell
# cluster_name="${cluster_name:-local}"
cluster_name="${cluster_name:-gpu3_64_docker}"
# cluster_name="${cluster_name:-gpu3_6_docker}"
model="${model:-resnet20}"
# model="${model:-resnet32}"
# model="${model:-resnet56}"
dataset="${dataset:-cifar10}"
# model="${model:-cifar10flnet}"
# dataset="${dataset:-cifar10}"
# model="${model:-efficientnet}"
# dataset="${dataset:-cifar100}"
partition_method="${partition_method:-noniid-#label2}"
# partition_method="${partition_method:-hetero}"
# partition_alpha="${partition_alpha:-0.1}" # 0.5
# model="${model:-mnistflnet}"
# dataset="${dataset:-mnist}"
algorithm="${algorithm:-CHOCO_SGD}"
# algorithm="${algorithm:-DPSGD}"
# algorithm="${algorithm:-FLOCK}"
# algorithm="${algorithm:-SAPS_FL}"
# algorithm="${algorithm:-FedAvg}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENTS_DIR="$( realpath "${SCRIPT_DIR}/.." )"
dir_name="${dir_name:-${EXPERIMENTS_DIR}}"

echo "dir_name--------: $dir_name"

source ${dir_name}/configs_system/$cluster_name.conf
source ${dir_name}/configs_model/$model.conf
source ${dir_name}/configs_algorithm/$algorithm.conf
source ${dir_name}/main_args.conf

main_args="${main_args:-  }"

# MPIRUN="/usr/bin/mpirun"
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/bin 
# MPIRUN="${MPIRUN:-mpirun}"
# MPIRUN="/usr/bin/mpirun"
# PYTHON="${PYTHON:-python}"
# PYTHON="${PYTHON:-python3}"
MPIRUN="${MPIRUN:-/usr/bin/mpirun}"
PYTHON=$(which python3)

MPI_ARGS="${MPI_ARGS:- -x PYENV_ROOT -x PATH}"

MPI_PROCESS="${MPI_PROCESS:-$PS_PROCESS}"
MPI_HOST="${MPI_HOST:-$PS_MPI_HOST}"

export WANDB_CONSOLE=off
echo $MPIRUN --allow-run-as-root $MPI_ARGS -np $MPI_PROCESS -host $MPI_HOST     $PYTHON ./main.py 
# echo ${PYTHON} ${MPIRUN},${MPI_ARGS},${MPI_PROCESS},${MPI_HOST},
# echo main_args $main_args,

# echo "MPIRU------------------------"
echo PYTHON $PYTHON
# $MPIRUN --allow-run-as-root $MPI_ARGS -np $MPI_PROCESS -host $MPI_HOST \
#     $PYTHON ./main.py $main_args \
if [[ "$DO_LAUNCH" == "1" ]]; then
    $MPIRUN --allow-run-as-root $MPI_ARGS -np $MPI_PROCESS -host $MPI_HOST \
        $PYTHON ./main.py $main_args
fi







