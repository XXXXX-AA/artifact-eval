import os, io, json, time, threading, errno, gc
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import torch
import logging

class SpillQueue:
    """
    每个 sender 一条持久化有序队列：
      - put(sender, round, payload): 落盘 + 更新索引
      - pop_next(sender): 读取 sender 的下一条（按序号），读完删文件
    线程安全：接收侧和训练侧可并发调用。
    """
    def __init__(self, self_rank):
        self.rank = self_rank
        self.base = Path.cwd() / "spill_queue" / str(self.rank)
        self.base.mkdir(parents=True, exist_ok=True)
        # NOTE: comment translated from Chinese
        self._next_seq: Dict[int, int] = {}
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        # NOTE: comment translated from Chinese
        self._ready_senders: set[int] = set()

        # NOTE: comment translated from Chinese
        self._rebuild_index()

    @staticmethod
    def _seq_name(seq: int) -> str:
        return f"{seq:012d}.pt"  # NOTE: comment translated from Chinese

    def _rebuild_index(self):
        """启动或崩溃恢复时从磁盘重建 next_seq 与 ready 集。"""
        for sender_dir in self.base.iterdir():
            if not sender_dir.is_dir():
                continue
            sender = int(sender_dir.name)
            # NOTE: comment translated from Chinese
            files = sorted(p for p in sender_dir.glob("*.pt"))
            if not files:
                continue
            first = int(files[0].stem)
            self._next_seq[sender] = first
            self._ready_senders.add(sender)

    def _sender_dir(self, sender: int) -> Path:
        p = self.base / str(sender)
        p.mkdir(parents=True, exist_ok=True)
        return p


    def pop_next(self, sender: int, block: bool = False, timeout: Optional[float] = None) -> Optional[Tuple[int, Any]]:
        """
        读取 sender 的下一条（按序号），返回 (seq, payload)；若无数据返回 None。
        block=True 时等待到来或超时。
        """
        deadline = None if timeout is None else time.time() + timeout
        with self._lock:
            while True:
                seq = self._next_seq.get(sender)
                if seq is None:
                    if not block:
                        return None
                    # NOTE: comment translated from Chinese
                    remaining = None if deadline is None else max(0, deadline - time.time())
                    if remaining == 0:
                        return None
                    self._cv.wait(timeout=remaining)
                    continue

                path = self._sender_dir(sender) / self._seq_name(seq)
                if path.exists():
                    # NOTE: comment translated from Chinese
                    payload = torch.load(path, map_location="cpu")
                    # NOTE: comment translated from Chinese
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    # NOTE: comment translated from Chinese
                    next_seq = None
                    # NOTE: comment translated from Chinese
                    pnext = self._sender_dir(sender) / self._seq_name(seq + 1)
                    if pnext.exists():
                        next_seq = seq + 1
                    else:
                        files = sorted(self._sender_dir(sender).glob("*.pt"))
                        next_seq = int(files[0].stem) if files else None

                    if next_seq is None:
                        self._next_seq.pop(sender, None)
                        self._ready_senders.discard(sender)
                    else:
                        self._next_seq[sender] = next_seq

                    # NOTE: comment translated from Chinese
                    gc.collect()
                    return (seq, payload)

                # NOTE: comment translated from Chinese
                if not block:
                    return None
                remaining = None if deadline is None else max(0, deadline - time.time())
                if remaining == 0:
                    return None
                self._cv.wait(timeout=remaining)

    def any_sender_ready(self) -> Optional[int]:
        """返回任意一个有数据的 sender（不消费），否则 None。"""
        with self._lock:
            return next(iter(self._ready_senders), None)

    def pop_next_from_any(self, block: bool = False, timeout: Optional[float] = None) -> Optional[Tuple[int, int, Any]]:
        """
        从任意 sender 读下一条，返回 (sender, seq, payload)。
        """
        deadline = None if timeout is None else time.time() + timeout
        with self._lock:
            while True:
                sid = self.any_sender_ready()
                if sid is None:
                    if not block:
                        return None
                    remaining = None if deadline is None else max(0, deadline - time.time())
                    if remaining == 0:
                        return None
                    self._cv.wait(timeout=remaining)
                    continue

                # NOTE: comment translated from Chinese
                sid = int(sid)
                break

        out = self.pop_next(sid, block=block, timeout=timeout)
        if out is None:
            return None
        seq, payload = out
        return (sid, seq, payload)

        # NOTE: comment translated from Chinese

    def _file_for(self, sender: int, seq: int) -> Path:
        """给定 sender 和 round(seq) 返回对应最终文件路径"""
        return self._sender_dir(int(sender)) / self._seq_name(int(seq))

    def exists(self, sender: int, sender_round: int) -> bool:
        """文件是否已落地可读（原子 rename 后才会返回 True）"""
        return self._file_for(sender, sender_round).exists()

    def peek_exact(self, sender: int, sender_round: int) -> Optional[Any]:
        """
        只读不删：若存在就读入并返回 payload；不存在返回 None。
        读取在 CPU 上进行（map_location='cpu'）。
        """
        p = self._file_for(sender, sender_round)
        if not p.exists():
            return None
        return torch.load(p, map_location="cpu")




    def put(self, sender: int, sender_round: int, payload: Any):
        with self._lock:
            d = self._sender_dir(sender)
            seq = int(sender_round)
            name = self._seq_name(seq)
            tmp = d / (name + ".tmp")
            fin = d / name
            ok  = d / (name + ".ok")  # NOTE: comment translated from Chinese

            # NOTE: comment translated from Chinese
            if fin.exists() and ok.exists():
                logging.debug(f"[spill] skip existing {fin}")
                return

            # NOTE: comment translated from Chinese
            with open(tmp, "wb") as f:
                # NOTE: comment translated from Chinese
                torch.save(payload, f)  # NOTE: comment translated from Chinese
                f.flush()
                os.fsync(f.fileno())

            os.rename(tmp, fin)
            # NOTE: comment translated from Chinese
            dirfd = os.open(d, os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)

            # NOTE: comment translated from Chinese
            with open(ok, "wb") as f:
                f.write(b"OK")
                f.flush()
                os.fsync(f.fileno())

            # NOTE: comment translated from Chinese
            dirfd = os.open(d, os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)


            # files_in_dir = [p.name for p in sorted(d.glob("*"))]
            # NOTE: comment translated from Chinese
            #     logging.info(f"[spill] sender={sender} round={sender_round} -> dir {d} contains: {files_in_dir}")

            # NOTE: comment translated from Chinese
            # next_seq = self._next_seq.get(sender)
            # if next_seq is None or seq < next_seq:
            #     self._next_seq[sender] = seq
            # self._ready_senders.add(sender)
            # self._cv.notify_all()




    def pop_exact(self, sender: int, sender_round: int,
              block: bool = False, timeout: Optional[float] = None) -> Optional[Any]:
        deadline = None if timeout is None else time.time() + timeout
        seq = int(sender_round)
        d   = self._sender_dir(sender)
        fin = d / self._seq_name(seq)
        ok  = d / (self._seq_name(seq) + ".ok")

        # NOTE: comment translated from Chinese
        while True:
            if fin.exists() and ok.exists():
                break
            if not block:
                return None
            remaining = None if deadline is None else max(0, deadline - time.time())
            if remaining == 0:
                return None
            time.sleep(min(0.05, remaining) if remaining is not None else 0.05)

        # NOTE: comment translated from Chinese
        def _magic_ok(p: Path) -> bool:
            try:
                with open(p, "rb") as f:
                    head = f.read(4)
                # NOTE: comment translated from Chinese
                if head.startswith(b"PK\x03\x04"):
                    return True
                # NOTE: comment translated from Chinese
                if len(head) >= 1 and head[0] == 0x80:
                    return True
                return False
            except Exception:
                return False

        retries = 5
        for _ in range(retries):
            if _magic_ok(fin):
                break
            time.sleep(0.05)
        else:
            # NOTE: comment translated from Chinese
            bad = d / (self._seq_name(seq) + ".corrupt")
            try:
                os.rename(fin, bad)
                if ok.exists():
                    os.remove(ok)
            except Exception:
                pass
            logging.warning("[spill] corrupt file detected: %s (moved to .corrupt)", fin)
            return None

        # NOTE: comment translated from Chinese
        payload = torch.load(fin, map_location="cpu")
        try:
            os.remove(fin)
        except OSError:
            pass
        try:
            os.remove(ok)
        except OSError:
            pass

        # NOTE: comment translated from Chinese
        with self._lock:
            nxt = self._next_seq.get(int(sender))
            if nxt is not None and seq == nxt:
                pnext = self._sender_dir(sender) / self._seq_name(nxt + 1)
                if pnext.exists() and (self._sender_dir(sender) / (self._seq_name(nxt + 1) + ".ok")).exists():
                    self._next_seq[int(sender)] = nxt + 1
                else:
                    files = sorted(self._sender_dir(sender).glob("*.pt"))
                    # NOTE: comment translated from Chinese
                    files = [x for x in files if (x.parent / (x.name + ".ok")).exists()]
                    if files:
                        self._next_seq[int(sender)] = int(files[0].stem)
                    else:
                        self._next_seq.pop(int(sender), None)
                        self._ready_senders.discard(int(sender))

        return payload


