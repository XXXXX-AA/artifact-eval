# GossipFL/utils/data_partition_io.py
import json, os, time
from pathlib import Path

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
        # NOTE: comment translated from Chinese
        pass

def save_partition_once(path, content):
    """
    原子写: 先写 .tmp，再 os.replace()
    """
    path = Path(path)
    # NOTE: comment translated from Chinese
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content))  # NOTE: comment translated from Chinese
    # NOTE: comment translated from Chinese
    _fsync_file_and_dir(tmp)

    os.replace(tmp, path)  # NOTE: comment translated from Chinese
    _fsync_file_and_dir(path)

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
