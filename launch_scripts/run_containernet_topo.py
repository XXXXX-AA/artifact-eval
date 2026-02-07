import os
import subprocess
import time
from pathlib import Path
from mininet.net import Containernet
from mininet.node import Controller, Docker
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI

SCRIPT_DIR = Path(__file__).resolve().parent
FLOCK_ROOT = Path(os.environ.get("FLOCK_ROOT", SCRIPT_DIR.parent)).resolve()
GOSSIPFL_DIR = Path(os.environ.get("GOSSIPFL_DIR", FLOCK_ROOT / "algorithm" / "GossipFL")).resolve()

def get_env_from_launch_script(launch_script):
    bash_command = f'''
    set -a
    export PROJECT_ROOT="/workspace/GossipFL"
    export DATA_ROOT="/workspace/GossipFL/data"
    export DO_LAUNCH=0
    source {launch_script}
    echo "NWORKERS=$NWORKERS"
    echo "main_args=\"$main_args\""
    '''
    result = subprocess.run(
        ['bash', '-c', bash_command],
        capture_output=True,
        text=True
    )
    env_vars = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            env_vars[k.strip()] = v.strip().strip('"')
    return env_vars

def create_network(nworkers, main_args, main_py_path):
    net = Containernet(controller=Controller)
    info('*** Adding controller\n')
    net.addController('c0')

    info('*** Adding docker containers\n')
    containers = []
    for i in range(nworkers):
        name = f'd{i+1}'
        container = net.addDocker(
            name=name,
            ip=f'10.0.0.{i+1}',
            dimage='gossipfl:latest',
            # dcmd=f'python3 experiments/mpi_based/main.py {main_args}',
            dcmd = f'python3 experiments/mpi_based/main.py --rank {i} --worker_number {nworkers} {main_args}',
            # volumes=["/path/to/workspace:/workspace"],
            volumes=[
                f"{GOSSIPFL_DIR}:/workspace/GossipFL"
            ],
            dargs={"runtime": "nvidia", "working_dir": "/workspace/GossipFL", "privileged": True}
        )
        containers.append(container)

    info('*** All containers created. Waiting 5 seconds...\n')
    time.sleep(5)

    info('*** Adding switch\n')
    switch = net.addSwitch('s1')

    info('*** Adding links\n')
    for container in containers:
        net.addLink(container, switch, cls=TCLink)

    info('*** Starting network\n')
    net.start()

    info('*** Running CLI\n')
    CLI(net)

    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    launch_script = str(GOSSIPFL_DIR / "experiments" / "mpi_based" / "launch_mpi_based_docker.sh")
    main_py = "/workspace/GossipFL/experiments/mpi_based/main.py"

    env = get_env_from_launch_script(launch_script)
    print("env.data_dir",env.get("data_dir"))
    nworkers_str = env.get("NWORKERS", "2").strip()
    if ":-" in nworkers_str:
        nworkers = int(nworkers_str.split(":-")[1].strip('}"'))
    else:
        try:
            nworkers = int(nworkers_str)
        except ValueError:
            nworkers = 2

    main_args = env.get("main_args", "")
    print(f"\nLaunching {nworkers} nodes, each runs: python3 main.py {main_args}\n")

    create_network(nworkers, main_args, main_py)
