"""
Thread Health Monitoring System

Provides thread health monitoring and alerting for background threads.
Each monitored thread should regularly send heartbeats. If a thread misses
heartbeats for too long, the monitor will raise an alert.
"""

import threading
import time
import logging
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ThreadStatus(Enum):
    """Thread health status"""
    HEALTHY = "healthy"
    STALE = "stale"  # Missed some heartbeats but not critical
    DEAD = "dead"    # Missed too many heartbeats, likely crashed


@dataclass
class ThreadHealth:
    """Health information for a monitored thread"""
    name: str
    last_heartbeat: float
    heartbeat_count: int
    status: ThreadStatus
    expected_interval: float  # Expected time between heartbeats (seconds)
    timeout_threshold: float  # Time without heartbeat before considered dead


class ThreadMonitor:
    """
    Central thread health monitor.

    Usage:
        monitor = ThreadMonitor()
        monitor.start()

        # In each background thread:
        while running:
            monitor.heartbeat("my_thread")
            # ... do work ...

        # Check health:
        status = monitor.get_status()
    """

    def __init__(self, check_interval: float = 1.0):
        """
        Args:
            check_interval: How often to check thread health (seconds)
        """
        self.check_interval = check_interval
        self._threads: Dict[str, ThreadHealth] = {}
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._alert_callbacks: Dict[str, Callable] = {}  # thread_name -> callback

    def register_thread(
        self,
        name: str,
        expected_interval: float = 1.0,
        timeout_threshold: float = 5.0,
        alert_callback: Optional[Callable] = None
    ):
        """
        Register a thread for monitoring.

        Args:
            name: Unique thread name
            expected_interval: Expected time between heartbeats (seconds)
            timeout_threshold: Time without heartbeat before considered dead
            alert_callback: Optional callback(name, status) to call on status change
        """
        with self._lock:
            self._threads[name] = ThreadHealth(
                name=name,
                last_heartbeat=time.time(),
                heartbeat_count=0,
                status=ThreadStatus.HEALTHY,
                expected_interval=expected_interval,
                timeout_threshold=timeout_threshold
            )
            if alert_callback:
                self._alert_callbacks[name] = alert_callback

        logger.info(f"Registered thread '{name}' for monitoring "
                   f"(interval={expected_interval}s, timeout={timeout_threshold}s)")

    def unregister_thread(self, name: str):
        """Unregister a thread from monitoring."""
        with self._lock:
            if name in self._threads:
                del self._threads[name]
                if name in self._alert_callbacks:
                    del self._alert_callbacks[name]
                logger.info(f"Unregistered thread '{name}'")

    def heartbeat(self, name: str):
        """
        Send a heartbeat from a monitored thread.

        Args:
            name: Thread name (must be registered first)
        """
        with self._lock:
            if name not in self._threads:
                logger.warning(f"Heartbeat from unregistered thread '{name}'")
                return

            thread = self._threads[name]
            thread.last_heartbeat = time.time()
            thread.heartbeat_count += 1

            # Reset status to healthy on heartbeat
            if thread.status != ThreadStatus.HEALTHY:
                old_status = thread.status
                thread.status = ThreadStatus.HEALTHY
                logger.info(f"Thread '{name}' recovered (was {old_status.value})")

    def get_thread_status(self, name: str) -> Optional[ThreadHealth]:
        """Get health status of a specific thread."""
        with self._lock:
            return self._threads.get(name)

    def get_all_status(self) -> Dict[str, ThreadHealth]:
        """Get health status of all monitored threads."""
        with self._lock:
            return dict(self._threads)

    def is_all_healthy(self) -> bool:
        """Check if all monitored threads are healthy."""
        with self._lock:
            return all(t.status == ThreadStatus.HEALTHY for t in self._threads.values())

    def start(self):
        """Start the health monitoring thread."""
        if self._running:
            logger.warning("Thread monitor already running")
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ThreadMonitor"
        )
        self._monitor_thread.start()
        logger.info("Thread monitor started")

    def stop(self):
        """Stop the health monitoring thread."""
        if not self._running:
            return

        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("Thread monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop that checks thread health."""
        while self._running:
            current_time = time.time()

            with self._lock:
                for name, thread in self._threads.items():
                    time_since_heartbeat = current_time - thread.last_heartbeat
                    old_status = thread.status

                    # Determine new status based on time since heartbeat
                    if time_since_heartbeat > thread.timeout_threshold:
                        thread.status = ThreadStatus.DEAD
                    elif time_since_heartbeat > thread.expected_interval * 2:
                        thread.status = ThreadStatus.STALE
                    else:
                        thread.status = ThreadStatus.HEALTHY

                    # Alert on status change
                    if thread.status != old_status:
                        self._alert_status_change(name, thread, old_status)

            time.sleep(self.check_interval)

    def _alert_status_change(self, name: str, thread: ThreadHealth, old_status: ThreadStatus):
        """Alert on thread status change."""
        logger.warning(
            f"Thread '{name}' status changed: {old_status.value} -> {thread.status.value} "
            f"(last heartbeat: {time.time() - thread.last_heartbeat:.1f}s ago)"
        )

        # Call registered callback if available
        if name in self._alert_callbacks:
            try:
                self._alert_callbacks[name](name, thread.status)
            except Exception as e:
                logger.error(f"Error in alert callback for thread '{name}': {e}")


# Global monitor instance (singleton pattern)
_global_monitor: Optional[ThreadMonitor] = None


def get_global_monitor() -> ThreadMonitor:
    """Get or create the global thread monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ThreadMonitor()
    return _global_monitor
