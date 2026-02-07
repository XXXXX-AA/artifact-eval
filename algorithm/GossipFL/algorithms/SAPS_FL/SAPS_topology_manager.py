import logging
import os, time, json
from pathlib import Path
import numpy as np

from fedml_core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager
from .minmax_commuication_cost import SAPS_gossip
from .utils import generate_bandwidth


class SAPSTopologyManager(SymmetricTopologyManager):
    """
    """

    def __init__(self, worker_index,args=None):
        super().__init__(n=args.client_num_in_total, neighbor_num=1)
        # super:
        # self.n = n
        # self.neighbor_num = neighbor_num
        # self.topology = []
        # self.args = args
        self.worker_index = worker_index
        self.bandwidth = generate_bandwidth(args)
        self.SAPS_gossip_match = SAPS_gossip(self.bandwidth, args.B_thres, args.T_thres)

    # override
    def generate_topology(self, t):
        rank = int(self.worker_index)
        current_dir = Path(__file__).parent
        out_dir = current_dir / "generate_bandwidth" / str(t)
        match_path = out_dir / "match.json"
        thr_path   = out_dir / "real_bandwidth_threshold.json"

        if rank == 0:
            raw_match, real_thr = self.SAPS_gossip_match.generate_match(t)
            logging.debug("rank0 generate_match(t=%s): type=%s", t, type(raw_match).__name__)

            # NOTE: comment translated from Chinese
            if isinstance(raw_match, dict):
                # NOTE: comment translated from Chinese
                match_list = [int(raw_match.get(i, raw_match.get(str(i)))) for i in range(self.n)]
            else:
                arr = np.asarray(raw_match).reshape(-1)
                if arr.size != self.n:
                    raise ValueError(f"match size mismatch: got {arr.size}, expect {self.n}")
                match_list = [int(x) for x in arr]

            _save_json_once(match_path, match_list)
            _save_json_once(thr_path, float(real_thr))
        else:
            match_list = _wait_and_load_json(match_path)
            real_thr   = _wait_and_load_json(thr_path)

        # NOTE: comment translated from Chinese
        match = np.asarray(match_list, dtype=np.int32)
        self.real_bandwidth_threshold = float(real_thr)

        logging.debug("match: %s", match.tolist())
        self.topology = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            self.topology[i][i] = 0.5
            self.topology[i][match[i]] = 0.5


    # NOTE: comment translated from Chinese
def _fsync_file_and_dir(path: Path):
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass

def _save_json_once(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content))
    os.replace(tmp, path)

def _wait_and_load_json(path: Path, retry=240, interval=0.25):
    path = Path(path)
    for _ in range(retry):
        if path.exists() and path.stat().st_size > 0:
            with open(path) as f:
                return json.load(f)
        time.sleep(interval)
    raise RuntimeError(f"file not found or empty: {path}")



