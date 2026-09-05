# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import atexit
import ctypes
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from torchspec.utils.logging import logger

if TYPE_CHECKING:
    # Imported lazily elsewhere in this package to avoid a circular dependency
    # with torchspec.config.mooncake_config.
    from torchspec.config.mooncake_config import MooncakeConfig


def get_free_port():
    # Create a standard TCP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Binding to '' (or 'localhost') and port 0 tells the OS to assign a free port
        s.bind(('', 0))
        # Retrieve the port number assigned by the OS
        return s.getsockname()[1]

def resolve_mooncake_master_bin() -> str:
    """Resolve the path to the mooncake_master binary."""
    if "MOONCAKE_BUILD_DIR" in os.environ:
        return os.path.join(os.environ["MOONCAKE_BUILD_DIR"], "mooncake-store/src/mooncake_master")

    which_result = shutil.which("mooncake_master")
    if which_result:
        return which_result

    home = os.path.expanduser("~")
    return os.path.join(home, "build/mooncake-store/src/mooncake_master")


def _subprocess_preexec():
    """Pre-exec setup for the mooncake master subprocess.

    - os.setpgrp(): Create a new process group so that os.killpg() can kill
      the wrapper script AND the real binary it spawns (grandchild).
    - PR_SET_PDEATHSIG: Kernel sends SIGTERM when the parent (Ray worker) dies,
      preventing orphans on crashes.
    """
    os.setpgrp()
    PR_SET_PDEATHSIG = 1
    ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


class MooncakeMaster:
    """Class that manages the mooncake master subprocess.

    Provides automatic lifecycle management — when the actor is killed or garbage
    collected, the subprocess is terminated. Logs are streamed through Ray's
    logging pipeline instead of written to files.
    """

    def __init__(self):
        self._process = None
        self._info = {}

    def start(
        self,
        port: int,
        host: str,
        http_port: int,
        http_host: str = "0.0.0.0",
        kv_lease_ttl_s: float = 5.0,
    ) -> dict:
        """Launch the mooncake master subprocess.

        Args:
            port: gRPC port for mooncake master.
            http_port: HTTP metadata server port.
            http_host: HTTP metadata server host.
            kv_lease_ttl_s: Default KV object lease TTL in seconds.

        Returns:
            Dict with "master_addr" and "metadata_port".

        Raises:
            FileNotFoundError: If binary is not found.
            RuntimeError: If process fails to start.
        """
        mooncake_bin = resolve_mooncake_master_bin()
        if not os.path.exists(mooncake_bin):
            raise FileNotFoundError(f"mooncake_master binary not found at {mooncake_bin}")

        # Pick a free port for metrics / admin server
        metrics_port = get_free_port()
        cmd = [
            mooncake_bin,
            f"--port={port}",
            f"--http_metadata_server_port={http_port}",
            f"--http_metadata_server_host={http_host}",
            "--enable_http_metadata_server=true",
            f"--default_kv_lease_ttl={int(kv_lease_ttl_s * 1000)}",
            f"--metrics_port={metrics_port}",
        ]

        logger.info(f"Starting mooncake master on port {port}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_subprocess_preexec,
        )

        # Stream stdout/stderr through logger in background daemon threads
        self._start_log_thread(self._process.stdout, "stdout")
        self._start_log_thread(self._process.stderr, "stderr")

        time.sleep(2)

        if self._process.poll() is not None:
            raise RuntimeError(
                f"mooncake master failed to start (exit code: {self._process.returncode})"
            )

        self._info = {
            "master_addr": f"{host}:{port}",
            "metadata_port": http_port,
        }

        logger.info(f"mooncake master started (PID: {self._process.pid})")
        return self._info

    def health_check(self) -> bool:
        """Check if the subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def get_info(self) -> dict:
        """Return the master address and metadata port."""
        return self._info

    def _start_log_thread(self, stream, name: str) -> None:
        """Start a daemon thread that reads lines from stream and logs them."""

        def _reader():
            for line in stream:
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.rstrip("\n")
                if line:
                    logger.debug(f"[mooncake_master {name}] {line}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()

    def shutdown(self):
        """Gracefully terminate the subprocess and its entire process group."""
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except Exception:
                    pass
            self._process = None

    def __del__(self):
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception:
                    pass


def check_mooncake_master_available(
    master_server_address: str,
    metadata_server: str,
    timeout: float = 5.0,
) -> None:
    """Verify mooncake master services are reachable.

    Probes the gRPC endpoint via TCP connect and the HTTP metadata endpoint
    so actors fail fast with a clear error before expensive model loading.

    Args:
        master_server_address: gRPC address, e.g. "10.1.2.3:50051".
        metadata_server: HTTP metadata URL, e.g. "http://10.1.2.3:8090/metadata".
        timeout: Per-probe connection timeout in seconds.

    Raises:
        RuntimeError: If either service is unreachable.
    """
    # gRPC port check (TCP connect)
    try:
        grpc_host, grpc_port_str = master_server_address.rsplit(":", 1)
        grpc_port = int(grpc_port_str)
    except ValueError as exc:
        raise RuntimeError(f"Invalid master_server_address {master_server_address!r}") from exc

    try:
        with socket.create_connection((grpc_host, grpc_port), timeout=timeout):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Mooncake master gRPC unreachable at {master_server_address}: {exc}"
        ) from exc

    # HTTP metadata server check (TCP connect to parsed host:port)
    parsed = urlparse(metadata_server)
    http_host = parsed.hostname
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if http_host is None:
        raise RuntimeError(f"Cannot parse host from metadata_server URL: {metadata_server!r}")

    try:
        with socket.create_connection((http_host, http_port), timeout=timeout):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Mooncake metadata server unreachable at {metadata_server}: {exc}"
        ) from exc

    logger.info(
        "Mooncake services reachable: master=%s, metadata=%s",
        master_server_address,
        metadata_server,
    )


def launch_mooncake_master(args):
    """Launch the mooncake master as a Ray actor.

    Auto-resolves master_server_address and metadata_port if not configured.
    When master_server_address specifies a host IP, pins the actor to that node so the
    mooncake master process starts on the intended machine.
    Writes resolved values back to args for downstream code.

    Args:
        args: Arguments namespace with mooncake_master_server_address, mooncake_metadata_port, etc.

    Returns:
        The MooncakeMasterActor handle, or None if binary not found.
    """

    master_addr = getattr(args, "mooncake_master_server_address", None)
    if master_addr is None:
        master_addr = os.environ.get("MOONCAKE_MASTER_SERVER_ADDRESS")
    if master_addr is None:
        logger.error(
                "Missing mooncake_master_server_address. "
                )
        raise ValueError("Missing mooncake_master_server_address in args.")

    if ":" in master_addr:
        host = master_addr.split(":")[0]
        port = int(master_addr.split(":")[1])
    else:
        host = master_addr
        port = getattr(args, "mooncake_master_port", 50051)

    http_port = getattr(args, "mooncake_metadata_port", None) or getattr(
        args, "mooncake_http_port", None
    )
    if http_port is None:
        raise ValueError("Missing mooncake_metadata_port in args.")
    else:
        args.mooncake_metadata_port = http_port
        logger.info(f"Auto-resolved mooncake metadata_port: {http_port}")
    http_host = getattr(args, "mooncake_http_host", "0.0.0.0")

    # Check binary existence before creating the actor
    mooncake_bin = resolve_mooncake_master_bin()
    if not os.path.exists(mooncake_bin):
        logger.warning(f"Binary not found at {mooncake_bin}, skipping launch")
        return None

    mooncake_master = MooncakeMaster()
    kv_lease_ttl_s = getattr(args, "mooncake_kv_lease_ttl_s", 5.0)

    try:
        mooncake_info = mooncake_master.start(port, host, http_port, http_host, kv_lease_ttl_s)
        # Write back resolved values (actor may have updated host from node IP)
        args.mooncake_master_server_address = mooncake_info["master_addr"]
        args.mooncake_metadata_port = mooncake_info["metadata_port"]
        logger.info(f"mooncake master server started: {mooncake_info}")
    except Exception as e:
        logger.error(f"Failed to launch mooncake master actor: {e}")
        return None

    def _cleanup():
        try:
            mooncake_master.shutdown()
        except Exception:
            pass

    atexit.register(_cleanup)

    return mooncake_master


@dataclass(frozen=True)
class MasterCapacity:
    """Snapshot of Mooncake master memory accounting, in bytes."""

    total_bytes: int
    allocated_bytes: int
    evicted_key_count: int
    evicted_bytes: int
    put_alloc_failures: int

    @property
    def available_bytes(self) -> int:
        return max(0, self.total_bytes - self.allocated_bytes)

    @property
    def usage_fraction(self) -> float:
        # Unknown capacity reads as full so callers fail closed.
        if self.total_bytes <= 0:
            return 1.0
        return self.allocated_bytes / self.total_bytes


_CAPACITY_METRICS = {
    "master_total_capacity_bytes": "total_bytes",
    "master_allocated_bytes": "allocated_bytes",
    "master_evicted_key_count": "evicted_key_count",
    "master_evicted_size_bytes": "evicted_bytes",
    "master_put_start_alloc_failures_total": "put_alloc_failures",
}


def _parse_prometheus_gauges(text: str, wanted: dict[str, str]) -> dict[str, int]:
    """Extract named scalar metrics from Prometheus text exposition format."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        name = name.split("{", 1)[0]
        field = wanted.get(name)
        if field is None or field in found:
            continue
        try:
            found[field] = int(float(value))
        except ValueError:
            continue
    return found


def fetch_master_metrics(
    master_server_address: str,
    metrics_port: int,
    timeout: float = 2.0,
) -> MasterCapacity | None:
    """Scrape memory accounting from the Mooncake master's HTTP metrics server.

    ``master_total_capacity_bytes`` aggregates every mounted segment, so the
    result covers the whole cluster regardless of how many clients contribute.

    Returns None if the master is unreachable or the capacity gauges are absent.
    Callers must treat None as "capacity unknown" and fall back to a
    conservative limit rather than assuming headroom.
    """
    host = master_server_address.rsplit(":", 1)[0]
    url = f"http://{host}:{metrics_port}/metrics"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # Callers poll this; returning None is the signal, so keep it quiet and
        # let them decide how loudly to complain.
        logger.debug("Mooncake metrics scrape failed at %s: %s", url, exc)
        return None

    values = _parse_prometheus_gauges(body, _CAPACITY_METRICS)
    if "total_bytes" not in values or "allocated_bytes" not in values:
        logger.debug("Mooncake metrics at %s missing capacity gauges", url)
        return None

    return MasterCapacity(
        total_bytes=values["total_bytes"],
        allocated_bytes=values["allocated_bytes"],
        evicted_key_count=values.get("evicted_key_count", 0),
        evicted_bytes=values.get("evicted_bytes", 0),
        put_alloc_failures=values.get("put_alloc_failures", 0),
    )


class MooncakeCapacityMonitor:
    """Cached view of Mooncake master memory usage.

    The master runs its own eviction at ``eviction_high_watermark_ratio``
    (default 0.90) and will drop objects consumers have not read yet, so callers
    should throttle below that. A rising ``evicted_key_count`` means it has
    already happened.
    """

    def __init__(
        self,
        config: "MooncakeConfig",
        cache_ttl: float = 0.5,
        timeout: float = 0.5,
        unreachable_log_interval: float = 30.0,
    ):
        self._config = config
        self._cache_ttl = cache_ttl
        # Scrapes are synchronous, and callers poll from an event loop, so the
        # timeout doubles as the worst-case stall.
        self._timeout = timeout
        self._unreachable_log_interval = unreachable_log_interval
        self._cached: MasterCapacity | None = None
        self._cached_at = 0.0
        self._have_cached = False
        self._last_evicted_key_count = 0
        self._last_unreachable_log = 0.0

    def snapshot(self) -> MasterCapacity | None:
        """Latest reading, or None if the master could not be scraped.

        Failures are cached alongside successes so an unreachable master costs
        one timeout per ``cache_ttl`` rather than one per call.
        """
        now = time.monotonic()
        if self._have_cached and now - self._cached_at < self._cache_ttl:
            return self._cached

        snapshot = fetch_master_metrics(
            self._config.master_server_address,
            self._config.metrics_port,
            timeout=self._timeout,
        )
        self._cached = snapshot
        self._cached_at = now
        self._have_cached = True

        if snapshot is None:
            self._warn_unreachable(now)
        else:
            self._warn_on_eviction(snapshot)
        return snapshot

    def _warn_unreachable(self, now: float) -> None:
        if now - self._last_unreachable_log < self._unreachable_log_interval:
            return
        self._last_unreachable_log = now
        logger.warning(
            "Mooncake master metrics unreachable at %s:%s — byte-level "
            "backpressure is inactive",
            self._config.master_server_address.rsplit(":", 1)[0],
            self._config.metrics_port,
        )

    def _warn_on_eviction(self, snapshot: MasterCapacity) -> None:
        if snapshot.evicted_key_count <= self._last_evicted_key_count:
            return
        logger.error(
            "Mooncake master evicted %d objects (%d cumulative, %.1f GiB) — "
            "samples may have been dropped before trainers read them",
            snapshot.evicted_key_count - self._last_evicted_key_count,
            snapshot.evicted_key_count,
            snapshot.evicted_bytes / (1024**3),
        )
        self._last_evicted_key_count = snapshot.evicted_key_count
