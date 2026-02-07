import json, os, time
from pathlib import Path
import numpy as np

path = str(Path(__file__).resolve().parent / "partitions" / "cifar10_part.json")
def wait_and_load_partition(path, retry=240, interval=0.25):
    """
    等待分区文件可读（存在且非空）再加载
    """
    path = Path(path)
    for _ in range(retry):
        try:
            if path.exists() and path.stat().st_size > 0:
                with open(path) as f:
                    return json.load(f)
        except Exception:
            pass
        time.sleep(interval)
    raise RuntimeError(f"partition file not found or empty: {path}")


for i in range(30):
    data = wait_and_load_partition(path)
    dataidxs = np.array(data[str(i)], dtype=np.int64)
    print(f"load {i} times")
    print(dataidxs)
