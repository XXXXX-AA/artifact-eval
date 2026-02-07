# -*- coding: utf-8 -*-
"""
SpillQueue: RAM-first + Disk-spill with bounded memory & background prefetch.

Env knobs (optional):
- SPILL_DIR        : base dir for spill files (default: ./spill-rank{rank})
- SPILL_RAM_MB     : RAM cache budget in MB (default: 128; 0 disables RAM cache)
- SPILL_RAM_TTL    : seconds for lazy flush from RAM to disk (default: 0 = disabled)
- SPILL_PREFETCH   : "1"/"0" to enable/disable background prefetch (default: "1")
"""
import os
import io
import gc
import time
import errno
import threading
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Set, List

import torch
import logging


def _now() -> float:
    return time.time()


class SpillQueue:
    """
    - put(sender, seq, payload): 优先放入 RAM（预算内不落盘）；预算不足时回退落盘(.pt+.ok)。
      * 即使走 RAM 分支，也会写一个很小的 `.ok` 标记，供上层“就绪检测”使用。
    - pop_exact(sender, seq) -> payload|None: 优先从 RAM 命中；否则从磁盘加载并删除(.pt 与 .ok)。
    - 预取线程：把磁盘上的就绪序号在预算内搬进 RAM（不删盘，直到消费）。
    - 可选延迟刷盘：RAM 中“存活超过 TTL”的条目异步写盘一份（仍保留在 RAM）。
    """

    def __init__(self, self_rank: int):
        self.rank = int(self_rank)

        # ---------- Paths ----------
        base_dir = os.environ.get("SPILL_DIR", f"./spill_queue/spill-rank{self.rank}")
        self.root: Path = Path(base_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        # Map: sender -> set(seq on disk with .ok present)
        self._present_on_disk: Dict[int, Set[int]] = {}

        # ---------- Concurrency ----------
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        # ---------- RAM cache (bounded by bytes) ----------
        try:
            mb_default = int(os.environ.get("SPILL_RAM_MB", "64"))
        except Exception:
            mb_default = 128
        self._ram_budget_bytes: int = max(0, mb_default) * 1024 * 1024
        self._ram_cache: Dict[Tuple[int, int], Any] = {}     # (sender, seq) -> payload
        self._ram_bytes: int = 0
        self._ram_insert_time: Dict[Tuple[int, int], float] = {}

        # NOTE: comment translated from Chinese
        self._ready_senders: Set[int] = set()

        # ---------- Prefetch & Lazy-flush ----------
        self._prefetch_stop = threading.Event()
        self._prefetch_th: Optional[threading.Thread] = None
        self._lazy_flush_th: Optional[threading.Thread] = None

        enable_prefetch = os.environ.get("SPILL_PREFETCH", "1") != "0"
        if enable_prefetch and self._ram_budget_bytes > 0:
            self._prefetch_th = threading.Thread(
                target=self._prefetch_loop, name=f"spill-prefetch-{self.rank}", daemon=True
            )
            self._prefetch_th.start()

        try:
            self._ram_ttl: float = float(os.environ.get("SPILL_RAM_TTL", "0"))
        except Exception:
            self._ram_ttl = 0.0
        if self._ram_ttl > 0.0:
            self._lazy_flush_th = threading.Thread(
                target=self._lazy_flush_loop, name=f"spill-lazyflush-{self.rank}", daemon=True
            )
            self._lazy_flush_th.start()

        # NOTE: comment translated from Chinese
        self._rebuild_index()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def put(self, sender: int, seq: int, payload: Any) -> None:
        """
        写入一条消息：RAM 预算内优先放 RAM（更快），并写 `.ok` 小标记；
        预算不足则落盘 .pt + .ok。
        """
        sender = int(sender)
        seq = int(seq)
        nbytes = self._payload_nbytes(payload)
        inserted_to_ram = False

        if self._ram_budget_bytes > 0 and nbytes <= self._ram_budget_bytes:
            with self._lock:
                if (self._ram_bytes + nbytes) <= self._ram_budget_bytes:
                    key = (sender, seq)
                    if key not in self._ram_cache:
                        self._ram_cache[key] = payload
                        self._ram_bytes += nbytes
                        self._ram_insert_time[key] = _now()
                        inserted_to_ram = True
                        # NOTE: comment translated from Chinese
                        ok = self._ok_of(sender, seq)
                        try:
                            self._atomic_write_bytes(ok, b"ok")
                        except Exception:
                            pass
                        sset = self._present_on_disk.setdefault(sender, set())
                        sset.add(seq)
                        self._ready_senders.add(sender)
                        self._cv.notify_all()

        if not inserted_to_ram:
            # NOTE: comment translated from Chinese
            self._write_to_disk(sender, seq, payload)
            with self._lock:
                sset = self._present_on_disk.setdefault(sender, set())
                sset.add(seq)
                self._ready_senders.add(sender)
                self._cv.notify_all()

    def pop_exact(self, sender: int, seq: int) -> Optional[Any]:
        """
        读取并消费指定 (sender, seq)。
        返回 payload（例如 dict）或 None（尚未到达/就绪）。
        - RAM 命中：直接返回，并删除 `.ok` 标记；更新 present 集合。
        - 未命中 RAM：若磁盘 .pt+.ok 齐备，加载到 CPU、删除两文件并返回。
        """
        sender = int(sender)
        seq = int(seq)
        key = (sender, seq)
        payload = None
        from_ram = False

        with self._lock:
            if key in self._ram_cache:
                payload = self._ram_cache.pop(key)
                self._ram_bytes -= self._payload_nbytes(payload)
                self._ram_insert_time.pop(key, None)
                from_ram = True

        if from_ram:
            # NOTE: comment translated from Chinese
            try:
                ok = self._ok_of(sender, seq)
                if ok.exists():
                    ok.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                sset = self._present_on_disk.get(sender)
                if sset is not None:
                    sset.discard(seq)
            if self.rank == 0:
                logging.info(f"SpillQueue: pop_exact from RAM: {sender}, seq: {seq}")
            return payload

        # NOTE: comment translated from Chinese
        p = self._path_of(sender, seq)
        ok = self._ok_of(sender, seq)
        if p.exists() and ok.exists():
            payload = torch.load(p, map_location="cpu")
            # NOTE: comment translated from Chinese
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                ok.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                sset = self._present_on_disk.get(sender)
                if sset is not None:
                    sset.discard(seq)
            gc.collect()
            if self.rank == 0:
                logging.info(f"SpillQueue: pop_exact from disk: {sender}, seq: {seq}")
            return payload

        return None

    def close(self, timeout: float = 1.0) -> None:
        """优雅关闭后台线程。"""
        self._prefetch_stop.set()
        if self._prefetch_th is not None:
            self._prefetch_th.join(timeout=timeout)
        if self._lazy_flush_th is not None:
            self._lazy_flush_th.join(timeout=timeout)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _sender_dir(self, sender: int) -> Path:
        d = self.root / f"{int(sender):06d}"
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        return d

    @staticmethod
    def _seq_name(seq: int) -> str:
        return f"{int(seq):012d}.pt"

    def _path_of(self, sender: int, seq: int) -> Path:
        return self._sender_dir(sender) / self._seq_name(seq)

    def _ok_of(self, sender: int, seq: int) -> Path:
        return self._sender_dir(sender) / (self._seq_name(seq) + ".ok")

    def _payload_nbytes(self, obj: Any) -> int:
        """
        粗略估算对象占用字节数：遍历 torch.Tensor；其他对象按常数近似。
        """
        def _nb(x: Any) -> int:
            if isinstance(x, dict):
                return sum(_nb(v) for v in x.values())
            if isinstance(x, (list, tuple)):
                return sum(_nb(v) for v in x)
            if torch.is_tensor(x):
                return x.element_size() * x.nelement()
            return 64  # NOTE: comment translated from Chinese
        try:
            return _nb(obj)
        except Exception:
            return 0

    def _atomic_write_bytes(self, path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _write_to_disk(self, sender: int, seq: int, payload: Any) -> None:
        """
        将 payload 写为 .pt 并生成 .ok 文件（原子替换确保完整性）。
        """
        path = self._path_of(sender, seq)
        ok = self._ok_of(sender, seq)
        # NOTE: comment translated from Chinese
        buf = io.BytesIO()
        torch.save(payload, buf)
        data = buf.getvalue()
        self._atomic_write_bytes(path, data)
        # touch ok
        self._atomic_write_bytes(ok, b"ok")

    # NOTE: comment translated from Chinese
    def _prefetch_loop(self) -> None:
        while not self._prefetch_stop.is_set():
            with self._lock:
                free = self._ram_budget_bytes - self._ram_bytes
                if free <= 0 or not self._present_on_disk:
                    candidates: List[Tuple[int, int]] = []
                else:
                    candidates = []
                    # NOTE: comment translated from Chinese
                    for sender, sset in self._present_on_disk.items():
                        if not sset:
                            continue
                        seq = min(sset)
                        key = (sender, seq)
                        if key in self._ram_cache:
                            continue
                        p = self._path_of(sender, seq)
                        ok = self._ok_of(sender, seq)
                        if p.exists() and ok.exists():
                            candidates.append((sender, seq))
            if not candidates:
                time.sleep(0.01)
                continue

            loaded: List[Tuple[int, int, Any, int]] = []
            for sender, seq in candidates:
                p = self._path_of(sender, seq)
                ok = self._ok_of(sender, seq)
                try:
                    if p.exists() and ok.exists():
                        payload = torch.load(p, map_location="cpu")
                        nbytes = self._payload_nbytes(payload)
                        loaded.append((sender, seq, payload, nbytes))
                except Exception:
                    continue

            with self._lock:
                for sender, seq, payload, nbytes in loaded:
                    if (self._ram_bytes + nbytes) > self._ram_budget_bytes:
                        break
                    key = (sender, seq)
                    if key in self._ram_cache:
                        continue
                    self._ram_cache[key] = payload
                    self._ram_bytes += nbytes
                    if self._ram_ttl > 0:
                        self._ram_insert_time[key] = _now()
            time.sleep(0.0)

    # NOTE: comment translated from Chinese
    def _lazy_flush_loop(self) -> None:
        ttl = self._ram_ttl
        while not self._prefetch_stop.is_set():
            now = _now()
            to_flush: List[Tuple[int, int]] = []
            with self._lock:
                for key, t0 in list(self._ram_insert_time.items()):
                    if now - t0 >= ttl:
                        to_flush.append(key)

            for sender, seq in to_flush:
                p = self._path_of(sender, seq)
                ok = self._ok_of(sender, seq)
                if p.exists() and ok.exists():
                    with self._lock:
                        self._ram_insert_time.pop((sender, seq), None)
                        sset = self._present_on_disk.setdefault(sender, set())
                        sset.add(seq)
                    continue
                # NOTE: comment translated from Chinese
                payload = None
                with self._lock:
                    payload = self._ram_cache.get((sender, seq))
                if payload is None:
                    with self._lock:
                        self._ram_insert_time.pop((sender, seq), None)
                    continue
                try:
                    self._write_to_disk(sender, seq, payload)
                    with self._lock:
                        self._ram_insert_time.pop((sender, seq), None)
                        sset = self._present_on_disk.setdefault(sender, set())
                        sset.add(seq)
                except Exception:
                    pass
            time.sleep(0.2)

    def _rebuild_index(self) -> None:
        try:
            for sdir in self.root.iterdir():
                if not sdir.is_dir():
                    continue
                try:
                    sender = int(sdir.name)
                except ValueError:
                    continue
                sset: Set[int] = set()
                for f in sdir.iterdir():
                    if f.suffix != ".pt":
                        continue
                    try:
                        seq = int(f.stem)
                    except ValueError:
                        continue
                    ok = sdir / (f.name + ".ok")
                    if ok.exists():
                        sset.add(seq)
                if sset:
                    self._present_on_disk[sender] = sset
        except FileNotFoundError:
            pass
        except Exception:
            # NOTE: comment translated from Chinese
            pass

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
