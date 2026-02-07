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
        """
        仅 rank0 生成 match 与 real_bandwidth_threshold 并写盘；
        其他 rank 等待写盘完成后读取；随后构建对称的 pair-wise 拓扑。
        """
        rank = int(self.worker_index)

        # NOTE: comment translated from Chinese
        current_dir = Path(__file__).parent
        out_dir = current_dir / "generate_bandwidth" / str(t)
        match_path = out_dir / "match.json"
        thr_path   = out_dir / "real_bandwidth_threshold.json"

        if rank == 0:
            # NOTE: comment translated from Chinese
            match, real_thr = self.SAPS_gossip_match.generate_match(t)
            logging.debug("rank0 generate_match(t=%s): %s", t, match)

            # NOTE: comment translated from Chinese
            match_list = list(map(int, np.asarray(match).reshape(-1)))  # NOTE: comment translated from Chinese
            _save_json_once(match_path, match_list)
            _save_json_once(thr_path, float(real_thr))
        else:
            # NOTE: comment translated from Chinese
            match_list = _wait_and_load_json(match_path)
            real_thr = _wait_and_load_json(thr_path)
        match = np.array(match_list)
        self.real_bandwidth_threshold = real_thr
        # match, self.real_bandwidth_threshold = self.SAPS_gossip_match.generate_match(t)
        logging.debug("match: %s" % match)
        self.topology = np.zeros([self.n, self.n])
        for i in range(self.n):
            self.topology[i][i] = 1/2
            self.topology[i][match[i]] = 1/2


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
    _fsync_file_and_dir(tmp)
    os.replace(tmp, path)
    _fsync_file_and_dir(path)

def _wait_and_load_json(path: Path, retry=240, interval=0.25):
    for _ in range(retry):
        try:
            if path.exists() and path.stat().st_size > 0:
                with open(path) as f:
                    return json.load(f)
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"file not found or empty: {path}")



