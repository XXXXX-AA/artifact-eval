import os, time, json, tempfile, threading
from queue import Queue
import torch

class SnapshotWriter:
    def __init__(self, out_dir, queue_size=8):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.q = Queue(maxsize=queue_size)
        self._stop = False
        self.th = threading.Thread(target=self._worker, daemon=True)
        self.th.start()

    def _atomic_write(self, bytes_or_path, target_path, is_tensor=False):
        
        fd, tmp = tempfile.mkstemp(dir=self.out_dir)
        os.close(fd)
        try:
            if is_tensor:
                torch.save(bytes_or_path, tmp)
            else:
                with open(tmp, "w") as f:
                    f.write(bytes_or_path)
            os.replace(tmp, target_path)
        finally:
            
            pass

    def _worker(self):
        while not self._stop:
            item = self.q.get()
            if item is None:
                break
            epoch, ts, snap_cpu, meta = item
            bin_path = os.path.join(self.out_dir, f"ep{epoch:04d}_{ts}.pt")
            meta_path = os.path.join(self.out_dir, f"ep{epoch:04d}_{ts}.json")
            
            self._atomic_write(snap_cpu, bin_path, is_tensor=True)
            del snap_cpu
            
            self._atomic_write(json.dumps(meta), meta_path, is_tensor=False)
            self.q.task_done()

    def enqueue(self, epoch:int, snap_cpu:torch.Tensor, meta:dict):
        ts = int(time.time())
        self.q.put((epoch, ts, snap_cpu, meta))
        return ts

    def stop(self):
        self.flush()
        self._stop = True
        self.q.put(None)
        self.th.join()

    def flush(self):
        """Block until all enqueued snapshots are fully written (bin + meta)."""
        self.q.join()



    # === scanning & reading utilities ===
import glob, re

# def _parse_epoch_from_name(name: str) -> int:
#     m = re.search(r"ep(\d+)_", name)
#     return int(m.group(1)) if m else -1

# class SnapshotReader:
#     """


#     """
#     def __init__(self, snap_dir: str):
#         self.snap_dir = snap_dir

#     def scan(self, pattern: str = "ep*.pt"):
#         """

#         """
#         paths = glob.glob(os.path.join(self.snap_dir, pattern))
#         def _parse_epoch(p):
#             m = re.search(r"ep(\d+)_", os.path.basename(p))
#             return int(m.group(1)) if m else -1
#         paths.sort(key=_parse_epoch)
#         # paths.sort(key=lambda p: _parse_epoch_from_name(os.path.basename(p)))
#         items = []
#         for p in paths:
#             items.append((_parse_epoch_from_name(os.path.basename(p)),
#                           p,
#                           os.path.splitext(p)[0] + ".json"))
#         return items

#     def read(self, bin_path: str, meta_path: str = None):
#         """



#         """
#         try:
#             flat_cpu = torch.load(bin_path, map_location="cpu", weights_only=True)
#         except TypeError:

#         meta = {}
#         if meta_path and os.path.exists(meta_path):
#             try:
#                 with open(meta_path, "r") as f:
#                     meta = json.load(f)
#             except Exception:
#                 pass
#         return flat_cpu, meta

