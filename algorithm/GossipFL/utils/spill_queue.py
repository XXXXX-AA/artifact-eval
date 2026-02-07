# -*- coding: utf-8 -*-
"""
SpillQueue: RAM-first + Disk-spill with bounded memory & background prefetch.

Features
- RAM-first: keep payloads in RAM within a bounded budget (and write a tiny .ok marker for readiness).
- Disk spill: when RAM budget is exceeded, stream-write to disk (.pt + .ok) using temp-file + atomic replace.
- Prefetch with "reserve": only prefetch from disk when enough free RAM budget is available.
- Optional lazy flush: after a TTL, also persist RAM-resident items to disk (still kept in RAM).
- Stats: optionally log put/pop/prefetch counters and current RAM usage periodically.
"""

# =========================
# One-stop configuration (you can edit these defaults directly)
# =========================
from typing import Optional

CFG_SPILL_DIR_BASE: Optional[str] = "./spill_queue/{rank}"  # Base spill dir; None = use env only
CFG_SPILL_RAM_MB: int = 0                                           # RAM budget in MB; 0 disables RAM cache
CFG_PREFETCH_ENABLED: bool = False                                     # Enable prefetch thread
CFG_PREFETCH_RESERVE: float = 0.6                                     # Prefetch only if free RAM >= this fraction
CFG_PUT_USE_FRAC: float = 0.7                                         # put() can consume at most this fraction
CFG_RAM_TTL_SEC: float = 0.0                                          # >0 enables lazy flush (seconds); 0 = off
CFG_STATS_EVERY: int = 0                                              # Log stats every N ops; 0 = disabled

# (All above can be overridden by environment variables:
#  SPILL_DIR, SPILL_RAM_MB, SPILL_PREFETCH, SPILL_PREFETCH_RESERVE,
#  SPILL_RAM_USE_FRAC_FOR_PUT, SPILL_RAM_TTL, SPILL_STATS_EVERY)

import os
import gc
import time
import errno
import threading
from pathlib import Path
from typing import Any, Dict, Tuple, Set, List

import torch
import logging


def _now() -> float:
    return time.time()


def _env_or(default_val, env_key: str, caster):
    """Return env value converted by `caster` if present; otherwise `default_val`."""
    try:
        if env_key in os.environ:
            return caster(os.environ[env_key])
    except Exception:
        pass
    return default_val


def _clamp01(x: float, default: float) -> float:
    """Clamp a value to [0, 1]; return `default` if conversion fails."""
    try:
        x = float(x)
        if 0.0 <= x <= 1.0:
            return x
        return default
    except Exception:
        return default


class SpillQueue:
    """
    - put(sender, seq, payload): Prefer placing payload in RAM (within budget) and write a tiny `.ok` marker
      so the upper layer can see readiness; otherwise stream-write to disk (.pt + .ok).
    - pop_exact(sender, seq) -> payload|None: Prefer returning the RAM-resident payload; if not found, load from
      disk and delete (.pt and .ok), then return the payload.
    - Prefetch thread: Only when free RAM >= reserve fraction, move the oldest ready disk items into RAM (do NOT
      delete files; they are removed only when consumed by pop_exact).
    - Optional lazy flush: If TTL > 0, persist RAM items to disk after TTL seconds (and still keep them in RAM).
    """

    def __init__(self, self_rank: int):
        self.rank = int(self_rank)

        # ---------- Configuration ----------
        base_dir_tpl = (
            _env_or(CFG_SPILL_DIR_BASE, "SPILL_DIR", str)
            if CFG_SPILL_DIR_BASE is not None
            else _env_or(None, "SPILL_DIR", str)
        )
        if base_dir_tpl is None:
            base_dir_tpl = "./spill_queue/spill-rank{rank}"
        base_dir = base_dir_tpl.format(rank=self.rank)
        self.root: Path = Path(base_dir)
        self.root.mkdir(parents=True, exist_ok=True)

        mb_default = _env_or(CFG_SPILL_RAM_MB, "SPILL_RAM_MB", int)
        self._ram_budget_bytes: int = max(0, int(mb_default)) * 1024 * 1024

        self._prefetch_enabled: bool = bool(
            _env_or(int(CFG_PREFETCH_ENABLED), "SPILL_PREFETCH", int) != 0
        )
        self._prefetch_reserve: float = _clamp01(
            _env_or(CFG_PREFETCH_RESERVE, "SPILL_PREFETCH_RESERVE", float),
            CFG_PREFETCH_RESERVE,
        )
        self._put_use_frac: float = _clamp01(
            _env_or(CFG_PUT_USE_FRAC, "SPILL_RAM_USE_FRAC_FOR_PUT", float),
            CFG_PUT_USE_FRAC,
        )
        self._ram_ttl: float = float(_env_or(CFG_RAM_TTL_SEC, "SPILL_RAM_TTL", float))
        self._stats_every: int = int(
            _env_or(CFG_STATS_EVERY, "SPILL_STATS_EVERY", int)
        )

        # ---------- Concurrency ----------
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        # ---------- RAM cache ----------
        self._ram_cache: Dict[Tuple[int, int], Any] = {}  # (sender, seq) -> payload
        self._ram_bytes: int = 0
        self._ram_insert_time: Dict[Tuple[int, int], float] = {}

        # Disk index: sender -> {seq} where .ok exists (ready items)
        self._present_on_disk: Dict[int, Set[int]] = {}
        # Ready senders (might have data in RAM or on disk)
        self._ready_senders: Set[int] = set()

        # ---------- Prefetch & Lazy-flush ----------
        self._prefetch_stop = threading.Event()
        self._prefetch_th: Optional[threading.Thread] = None
        self._lazy_flush_th: Optional[threading.Thread] = None

        if self._prefetch_enabled and self._ram_budget_bytes > 0:
            self._prefetch_th = threading.Thread(
                target=self._prefetch_loop,
                name=f"spill-prefetch-{self.rank}",
                daemon=True,
            )
            self._prefetch_th.start()

        if self._ram_ttl > 0.0:
            self._lazy_flush_th = threading.Thread(
                target=self._lazy_flush_loop,
                name=f"spill-lazyflush-{self.rank}",
                daemon=True,
            )
            self._lazy_flush_th.start()

        # Stats (optional)
        self.stats_put_ram = 0
        self.stats_put_disk = 0
        self.stats_pop_ram = 0
        self.stats_pop_disk = 0
        self.stats_prefetch = 0
        self._ops = 0

        # Rebuild disk index at startup
        self._rebuild_index()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def put(self, sender: int, seq: int, payload: Any) -> None:
        """
        Insert a message:
        - If within RAM budget, keep it in RAM and write a small `.ok` marker for readiness detection.
        - Otherwise, stream-write it to disk (.pt + .ok).
        """
        sender = int(sender)
        seq = int(seq)
        nbytes = self._payload_nbytes(payload)
        inserted_to_ram = False

        if self._ram_budget_bytes > 0 and nbytes <= self._ram_budget_bytes:
            with self._lock:
                # Only allow put() to occupy up to _put_use_frac of the RAM budget
                if (self._ram_bytes + nbytes) <= self._ram_budget_bytes * self._put_use_frac:
                    key = (sender, seq)
                    if key not in self._ram_cache:
                        self._ram_cache[key] = payload
                        self._ram_bytes += nbytes
                        self._ram_insert_time[key] = _now()
                        inserted_to_ram = True
                        # Even in RAM path, write a tiny `.ok` marker so upper layers "see" readiness
                        ok = self._ok_of(sender, seq)
                        self._atomic_write_ok(ok)
                        sset = self._present_on_disk.setdefault(sender, set())
                        sset.add(seq)
                        self._ready_senders.add(sender)
                        self._cv.notify_all()
                        self._stat_inc("put_ram")

        if not inserted_to_ram:
            # Disk path: stream-write .pt using temp file + atomic replace, then write .ok
            self._write_to_disk(sender, seq, payload)
            with self._lock:
                sset = self._present_on_disk.setdefault(sender, set())
                sset.add(seq)
                self._ready_senders.add(sender)
                self._cv.notify_all()
                self._stat_inc("put_disk")

    def pop_exact(self, sender: int, seq: int) -> Optional[Any]:
        """
        Consume the specified (sender, seq).
        Returns payload (e.g., dict) or None if not ready yet.

        - If RAM hit: return payload and delete `.ok` marker; update internal index.
        - If not in RAM but .pt + .ok exist on disk: load from disk, delete both files, return payload.
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
            # RAM hit: also remove `.ok` and update index
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
                self._stat_inc("pop_ram")
            self._maybe_trim_os_memory()
            return payload

        # Disk path
        p = self._path_of(sender, seq)
        ok = self._ok_of(sender, seq)
        if p.exists() and ok.exists():
            payload = torch.load(p, map_location="cpu")
            # Delete .pt and .ok
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
                self._stat_inc("pop_disk")
            gc.collect()
            self._maybe_trim_os_memory()
            return payload

        return None

    def close(self, timeout: float = 1.0) -> None:
        """Gracefully stop background threads."""
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
        Roughly estimate RAM bytes: sum tensor bytes; use a small constant for non-tensors/metadata.
        """
        def _nb(x: Any) -> int:
            if isinstance(x, dict):
                return sum(_nb(v) for v in x.values())
            if isinstance(x, (list, tuple)):
                return sum(_nb(v) for v in x)
            if torch.is_tensor(x):
                return x.element_size() * x.nelement()
            return 64  # coarse constant for non-tensors / overhead
        try:
            return _nb(obj)
        except Exception:
            return 0

    def _atomic_write_ok(self, ok_path: Path) -> None:
        """Atomically write a small `.ok` file."""
        tmp = ok_path.with_suffix(ok_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(b"ok")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ok_path)

    def _write_to_disk(self, sender: int, seq: int, payload: Any) -> None:
        """
        Stream-write payload to .pt and create .ok.
        Use temp file + atomic replace to avoid a large in-memory BytesIO copy.
        """
        path = self._path_of(sender, seq)
        ok = self._ok_of(sender, seq)

        # Write .pt to a temp file, then atomically replace
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

        # Write .ok
        self._atomic_write_ok(ok)

    # Prefetch oldest ready seq from disk into RAM (within budget).
    # Do NOT delete files here; deletion happens only when consumed by pop_exact.
    def _prefetch_loop(self) -> None:
        while not self._prefetch_stop.is_set():
            with self._lock:
                free = self._ram_budget_bytes - self._ram_bytes
                # Only prefetch if free budget >= reserve fraction
                if (not self._present_on_disk) or (free < self._prefetch_reserve * self._ram_budget_bytes):
                    candidates: List[Tuple[int, int]] = []
                else:
                    candidates = []
                    for sender, sset in self._present_on_disk.items():
                        if not sset:
                            continue
                        seq = min(sset)  # pick the oldest seq
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
                added = 0
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
                    added += 1
                if added:
                    self.stats_prefetch += added
                    self._maybe_log_stats_unlocked()
            time.sleep(0.0)

    # Lazy flush: for RAM items alive longer than TTL, persist them to disk (still keep them in RAM).
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
        """Rebuild disk index (present_on_disk) by scanning .pt files that have a matching .ok."""
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
            # Non-fatal: the index will correct itself as new writes/reads happen
            pass

    # -----------------------------------------------------------------------
    # Stats & Utilities
    # -----------------------------------------------------------------------
    def _stat_inc(self, which: str) -> None:
        if which == "put_ram":
            self.stats_put_ram += 1
        elif which == "put_disk":
            self.stats_put_disk += 1
        elif which == "pop_ram":
            self.stats_pop_ram += 1
        elif which == "pop_disk":
            self.stats_pop_disk += 1
        self._ops += 1
        if self._stats_every > 0 and (self._ops % self._stats_every == 0):
            self._maybe_log_stats_unlocked()

    def _maybe_log_stats_unlocked(self) -> None:
        # Only print on rank 0 to avoid log storms
        if self.rank != 0 or self._stats_every <= 0:
            return
        logging.info(
            "[spill stats] put_ram=%d put_disk=%d pop_ram=%d pop_disk=%d "
            "prefetch_loads=%d ram_bytes=%.1fMB budget=%.1fMB reserve=%.0f%% put_frac=%.0f%%",
            self.stats_put_ram,
            self.stats_put_disk,
            self.stats_pop_ram,
            self.stats_pop_disk,
            self.stats_prefetch,
            self._ram_bytes / 1024 / 1024,
            self._ram_budget_bytes / 1024 / 1024,
            self._prefetch_reserve * 100,
            self._put_use_frac * 100,
        )

    def _maybe_trim_os_memory(self):
        """Try to return unused heap to the OS (glibc); best-effort, may do nothing on some platforms."""
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
