#!/usr/bin/env python3
import sys
import os
from pathlib import Path
sys.path = [p for p in sys.path if 'containernet' not in p]

# from mininet.link import TCLink
# from mininet.node import Controller
# from containernet.net import Containernet
from mininet.net import Containernet
from mininet.link import TCLink
from mininet.node import Controller

SCRIPT_DIR = Path(__file__).resolve().parent
FLOCK_ROOT = Path(os.environ.get("FLOCK_ROOT", SCRIPT_DIR.parent)).resolve()
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", FLOCK_ROOT)).resolve()

if __name__ == '__main__':
    net = Containernet(controller=Controller, link=TCLink)
    # Add an OpenFlow controller (optional)
    net.addController('c0')

    # Add two Docker hosts using the gossipfl:latest image and nvidia runtime
    d1 = net.addDocker(
        'd1',
        ip='10.0.0.251/24',
        dimage='gossipfl:latest',
        dargs={'runtime':'nvidia','volumes':[f'{WORKSPACE_DIR}:/workspace:rw']}
    )
    d2 = net.addDocker(
        'd2',
        ip='10.0.0.252/24',
        dimage='gossipfl:latest',
        dargs={'runtime':'nvidia','volumes':[f'{WORKSPACE_DIR}:/workspace:rw']}
    )

    # Rate-limit with TCLink: 1 Mbps and 100 ms delay
    net.addLink(d1, d2, bw=1, delay='100ms')

    net.start()
    print("--- testing connectivity ---")
    net.ping([d1, d2])
    # Start shells or run training scripts in the containers
    net.get('d1').cmd('bash &')  
    net.get('d2').cmd('bash &')

    # Enter CLI
    from mininet.cli import CLI
    CLI(net)

    net.stop()
