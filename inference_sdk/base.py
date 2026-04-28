"""
Base Inference Engine with LeRobot-style Async Inference.

Key Features (LeRobot Pattern):
- Timestamp-aligned action queue (not FIFO)
- Time-based action selection (skip expired actions)
- Adaptive chunk threshold based on latency estimation
- Observation queue maxsize=1 (always use latest frame)
- Aggregate function for overlapping action chunks
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from .monitor import get_global_monitor

logger = logging.getLogger(__name__)

@dataclass
class TraceEvent:
    timestamp: float
    source: str
    event: str
    details: Dict[str, Any] = field(default_factory=dict)

class TraceRecorder:
    """
    Simple recorder for tracing async inference events.

    Memory-safe: Limits event history to prevent unbounded growth
    during long-running inference sessions.
    """
    def __init__(self, max_events: int = 1000):
        """
        Args:
            max_events: Maximum number of events to keep (default: 1000)
                       Older events are automatically discarded.
        """
        self.events: List[TraceEvent] = []
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._max_events = max_events

    def record(self, source: str, event: str, **details):
        with self._lock:
            self.events.append(TraceEvent(
                timestamp=time.time() - self._start_time,
                source=source,
                event=event,
                details=details
            ))

            # Limit event history to prevent memory leak
            if len(self.events) > self._max_events:
                # Remove oldest 10% when limit exceeded
                remove_count = self._max_events // 10
                self.events = self.events[remove_count:]

    def clear(self):
        with self._lock:
            self.events = []
            self._start_time = time.time()

# ==================== Gripper Smoothing ====================
# ==================== Aggregate Functions ====================
# Following LeRobot pattern: configs.py

AGGREGATE_FUNCTIONS = {
    "latest_only": lambda old, new: new,
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


# ==================== Data Structures (LeRobot Pattern) ====================

@dataclass
class TimedAction:
    """Action with timestamp for time-aligned execution."""
    timestamp: float  # Absolute time when this action should be executed
    timestep: int     # Sequential step index
    action: np.ndarray
    
    def get_timestamp(self) -> float:
        return self.timestamp
    
    def get_timestep(self) -> int:
        return self.timestep
    
    def get_action(self) -> np.ndarray:
        return self.action


@dataclass  
class TimedObservation:
    """Observation with timestamp for latency tracking."""
    timestamp: float  # When observation was captured
    timestep: int     # Sequential step index
    images: Dict[str, np.ndarray]
    state: np.ndarray
    must_go: bool = False  # If True, this observation MUST be processed
    
    def get_timestamp(self) -> float:
        return self.timestamp
    
    def get_timestep(self) -> int:
        return self.timestep


@dataclass
class SmoothingConfig:
    """Configuration for async inference and action smoothing."""
    # Control frequency
    control_fps: float = 30.0
    
    # Gripper velocity clamping (in raw action space, [0, 1000])
    gripper_max_velocity: float = 200.0
    enable_gripper_clamping: bool = True
    
    # Async inference settings
    enable_async_inference: bool = True
    
    # Chunk threshold: trigger new inference when queue_size / chunk_size <= threshold
    # This is adaptive based on latency
    chunk_size_threshold: float = 0.5
    
    # Latency estimation
    latency_ema_alpha: float = 0.2  # EMA smoothing factor
    latency_safety_margin: float = 1.5  # Multiply latency estimate by this for safety
    
    # Aggregate function for overlapping chunks
    aggregate_fn_name: str = "latest_only"  # "latest_only", "weighted_average", etc.
    
    # Observation queue settings (LeRobot uses maxsize=1)
    obs_queue_maxsize: int = 1
    
    # Fallback when queue empty
    fallback_mode: str = "repeat"  # "repeat", "hold"
    
    # Legacy compatibility flags (not used in new implementation)
    fully_async: bool = False  # Ignored, always use async if enable_async_inference=True
    enable_async_prefetch: bool = True  # Mapped to enable_async_inference
    
    @property
    def environment_dt(self) -> float:
        """Time step in seconds."""
        return 1.0 / self.control_fps


# ==================== Latency Estimator ====================

class LatencyEstimator:
    """Estimates inference latency using exponential moving average."""
    
    def __init__(self, alpha: float = 0.2, initial_value: float = 0.1):
        self.alpha = alpha
        self.value = initial_value
        self._initialized = False
        self._lock = threading.Lock()
    
    def update(self, latency: float):
        """Update estimate with new measurement."""
        with self._lock:
            if not self._initialized:
                self.value = latency
                self._initialized = True
            else:
                self.value = self.alpha * latency + (1 - self.alpha) * self.value
    
    def get_value(self) -> float:
        """Get current estimate."""
        with self._lock:
            return self.value
    
    def get_steps_during_inference(self, fps: float) -> int:
        """Calculate how many control steps will pass during one inference."""
        with self._lock:
            return int(np.ceil(self.value * fps))


# ==================== Action Queue Manager (LeRobot Pattern) ====================

class TimestampedActionQueue:
    """
    LeRobot-style action queue with timestamp alignment.

    Key differences from simple deque:
    1. Actions are indexed by timestep, not FIFO order
    2. Supports aggregation when new chunk overlaps with existing actions
    3. Time-based action retrieval (skip expired actions)
    4. Thread-safe operations

    Performance optimization: maintains a sorted timestep list using bisect
    to avoid O(n log n) sorting on every get_action_for_time() call.
    """

    def __init__(self, config: SmoothingConfig):
        self.config = config
        self._queue: Dict[int, TimedAction] = {}  # timestep -> TimedAction
        self._sorted_timesteps: List[int] = []  # Sorted list of timesteps for O(log n) lookup
        self._lock = threading.Lock()
        self._latest_executed_timestep: int = -1
        self._chunk_size: int = 1
        self._aggregate_fn = get_aggregate_function(config.aggregate_fn_name)
    
    def reset(self):
        """Reset queue state for new episode."""
        with self._lock:
            self._queue.clear()
            self._sorted_timesteps.clear()
            self._latest_executed_timestep = -1
    
    def set_chunk_size(self, size: int):
        """Set expected chunk size (for threshold calculation)."""
        self._chunk_size = max(1, size)
    
    def get_queue_size(self) -> int:
        """Get number of actions in queue."""
        with self._lock:
            return len(self._queue)
    
    def get_fill_ratio(self) -> float:
        """Get queue fill ratio relative to chunk size."""
        with self._lock:
            return len(self._queue) / max(1, self._chunk_size)
    
    def should_request_new_chunk(self, latency_estimator: LatencyEstimator) -> bool:
        """
        Determine if we should trigger new inference.
        
        LeRobot pattern: trigger when queue_size / chunk_size <= threshold
        BUT also consider: we need enough actions to cover inference latency
        """
        with self._lock:
            queue_size = len(self._queue)
        
        # Basic threshold check
        fill_ratio = queue_size / max(1, self._chunk_size)
        if fill_ratio > self.config.chunk_size_threshold:
            return False
        
        # Latency-aware check: do we have enough actions to survive inference?
        steps_during_inference = latency_estimator.get_steps_during_inference(self.config.control_fps)
        safety_steps = int(steps_during_inference * self.config.latency_safety_margin)
        
        return queue_size <= safety_steps
    
    def add_action_chunk(self, timed_actions: List[TimedAction]):
        """
        Add new action chunk with aggregation for overlapping timesteps.

        LeRobot pattern (from robot_client.py _aggregate_action_queues):
        - Skip actions older than latest executed
        - Aggregate overlapping timesteps
        - Add new timesteps directly
        """
        import bisect

        with self._lock:
            for new_action in timed_actions:
                timestep = new_action.get_timestep()

                # Skip actions older than what we've already executed
                if timestep <= self._latest_executed_timestep:
                    continue

                # Check if this timestep already exists
                if timestep in self._queue:
                    # Aggregate with existing action (timestep already in sorted list)
                    old_action = self._queue[timestep].get_action()
                    aggregated = self._aggregate_fn(old_action, new_action.get_action())
                    self._queue[timestep] = TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=timestep,
                        action=aggregated
                    )
                else:
                    # Add new action directly and maintain sorted timesteps
                    self._queue[timestep] = new_action
                    bisect.insort(self._sorted_timesteps, timestep)

            logger.debug(f"Queue updated: {len(self._queue)} actions, "
                        f"latest_executed={self._latest_executed_timestep}")
    
    def get_action_for_time(self, current_time: float, t0: float) -> Optional[TimedAction]:
        """
        Get action for current time using timestamp alignment.

        LeRobot pattern: calculate which timestep we SHOULD be at,
        then get that action (or nearest future one).

        Args:
            current_time: Current wall clock time
            t0: Episode start time

        Returns:
            TimedAction for current timestep, or None if queue empty
        """
        import bisect

        with self._lock:
            if not self._queue:
                return None

            # Calculate expected timestep based on elapsed time
            elapsed = current_time - t0
            expected_timestep = int(elapsed / self.config.environment_dt)

            # Find the action to execute using binary search on sorted timesteps:
            # 1. If expected_timestep exists, use it
            # 2. If not, use the smallest timestep > latest_executed
            # 3. Skip any timesteps < expected_timestep (they're expired)

            # Use binary search to find first timestep > latest_executed
            idx = bisect.bisect_right(self._sorted_timesteps, self._latest_executed_timestep)

            if idx >= len(self._sorted_timesteps):
                return None

            # Get valid timesteps (those after latest_executed)
            valid_timesteps = self._sorted_timesteps[idx:]

            if not valid_timesteps:
                return None

            # Try to find expected_timestep or the next available one
            target_timestep = None
            for ts in valid_timesteps:
                if ts >= expected_timestep:
                    target_timestep = ts
                    break

            # If all valid timesteps are before expected, use the latest one
            # (this means we're behind, but at least we're moving forward)
            if target_timestep is None:
                target_timestep = valid_timesteps[-1]

            # Get action and update state
            action = self._queue.pop(target_timestep)
            self._sorted_timesteps.remove(target_timestep)
            self._latest_executed_timestep = target_timestep

            # Clean up any expired actions we skipped
            expired = [ts for ts in list(self._sorted_timesteps) if ts < target_timestep]
            for ts in expired:
                del self._queue[ts]
                self._sorted_timesteps.remove(ts)
                logger.debug(f"Discarded expired action timestep {ts}")

            return action
    
    def get_next_action(self) -> Optional[TimedAction]:
        """
        Simple FIFO-style get (fallback when not using timestamp alignment).
        Gets the action with smallest timestep > latest_executed.
        """
        import bisect

        with self._lock:
            if not self._queue:
                return None

            # Use binary search to find first timestep > latest_executed
            idx = bisect.bisect_right(self._sorted_timesteps, self._latest_executed_timestep)

            if idx >= len(self._sorted_timesteps):
                return None

            target_timestep = self._sorted_timesteps[idx]
            action = self._queue.pop(target_timestep)
            self._sorted_timesteps.remove(target_timestep)
            self._latest_executed_timestep = target_timestep

            return action


# ==================== Observation Queue (LeRobot Pattern) ====================

class ObservationQueue:
    """
    LeRobot-style observation queue with maxsize=1.
    
    Key insight: GPU inference is slower than camera capture.
    If we queue observations, we're always processing stale data.
    Solution: Only keep the LATEST observation, discard old ones.
    """
    
    def __init__(self, maxsize: int = 1):
        self._queue: Queue = Queue(maxsize=maxsize)
        self._lock = threading.Lock()
    
    def put(self, obs: TimedObservation) -> bool:
        """
        Add observation, discarding old one if queue is full.
        
        LeRobot pattern (from policy_server.py _enqueue_observation):
        If queue is full, pop the old observation to make room.
        """
        with self._lock:
            if self._queue.full():
                try:
                    _ = self._queue.get_nowait()
                    logger.debug("Observation queue full, discarded oldest")
                except Empty:
                    pass
            
            try:
                self._queue.put_nowait(obs)
                return True
            except Full:
                return False
    
    def get(self, timeout: float = 0.1) -> Optional[TimedObservation]:
        """Get observation with timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_nowait(self) -> Optional[TimedObservation]:
        """Get observation without blocking."""
        try:
            return self._queue.get_nowait()
        except Empty:
            return None
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def clear(self):
        """Clear all observations."""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Empty:
                    break


# ==================== Gripper Smoother ====================

class GripperSmoother:
    """Velocity clamping for gripper to prevent jerky movements."""
    
    def __init__(self, config: SmoothingConfig, action_dim: int = 7):
        self.config = config
        self.action_dim = action_dim
        self._last_action: Optional[np.ndarray] = None
    
    def reset(self):
        """Reset state for new episode."""
        self._last_action = None
    
    def smooth(self, action: np.ndarray) -> np.ndarray:
        """Apply velocity clamping to gripper."""
        if not self.config.enable_gripper_clamping or self._last_action is None:
            self._last_action = action.copy()
            return action
        
        result = action.copy()
        
        # Clamp gripper (last dimension)
        if len(action) >= 7:
            gripper_idx = -1
            delta = action[gripper_idx] - self._last_action[gripper_idx]
            clamped_delta = np.clip(
                delta, 
                -self.config.gripper_max_velocity,
                self.config.gripper_max_velocity
            )
            result[gripper_idx] = self._last_action[gripper_idx] + clamped_delta
        
        self._last_action = result.copy()
        return result
    
    def get_last_action(self) -> Optional[np.ndarray]:
        """Get last action (for fallback)."""
        return self._last_action.copy() if self._last_action is not None else None


# ==================== Async Inference Engine ====================

class AsyncInferenceWorker:
    """
    Background inference worker thread.
    
    LeRobot pattern:
    - Observation queue maxsize=1 (always latest)
    - Continuous inference in background
    - Results go to action queue with timestamp
    """
    
    def __init__(
        self,
        config: SmoothingConfig,
        inference_fn: Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray],
        action_queue: TimestampedActionQueue,
        latency_estimator: LatencyEstimator,
        trace_recorder: Optional[TraceRecorder] = None,
    ):
        self.config = config
        self._inference_fn = inference_fn
        self._action_queue = action_queue
        self._latency_estimator = latency_estimator
        self._trace_recorder = trace_recorder
        
        self._obs_queue = ObservationQueue(maxsize=config.obs_queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._must_go_event = threading.Event()
        self._must_go_event.set()  # Initially set
    
    def start(self):
        """Start background inference thread."""
        if self._running:
            return

        self._running = True

        # Register with thread monitor
        monitor = get_global_monitor()
        monitor.register_thread(
            name="AsyncInferenceWorker",
            expected_interval=2.0,  # Inference may take time, allow slack
            timeout_threshold=10.0  # Consider dead if no heartbeat for 10s
        )

        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AsyncInferenceWorker"
        )
        self._thread.start()
        logger.info("Async inference worker started")
    
    def stop(self):
        """Stop background inference thread."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._obs_queue.clear()

        # Unregister from thread monitor
        monitor = get_global_monitor()
        monitor.unregister_thread("AsyncInferenceWorker")

        logger.info("Async inference worker stopped")
    
    def submit_observation(self, obs: TimedObservation):
        """
        Submit observation for inference.
        
        LeRobot pattern:
        - If queue empty and must_go set, mark observation as must_go
        - must_go observations bypass similarity checks
        """
        # Check if we need this observation processed urgently
        if self._must_go_event.is_set() and self._action_queue.get_queue_size() == 0:
            obs.must_go = True
            self._must_go_event.clear()
            if self._trace_recorder:
                self._trace_recorder.record("Worker", "MustGo Triggered", timestep=obs.get_timestep())
        
        self._obs_queue.put(obs)
    
    def _worker_loop(self):
        """Background worker loop."""
        monitor = get_global_monitor()

        while self._running:
            # Send heartbeat at start of each iteration
            monitor.heartbeat("AsyncInferenceWorker")

            try:
                # Get observation (with timeout to allow clean shutdown)
                obs = self._obs_queue.get(timeout=0.1)
                if obs is None:
                    continue
                
                # Check if we should process this observation
                should_process = self._should_process_observation(obs)
                
                if not should_process:
                    logger.debug(f"Skipping observation timestep {obs.get_timestep()}")
                    if self._trace_recorder:
                        self._trace_recorder.record("Worker", "Skipped Obs", 
                                                  timestep=obs.get_timestep(),
                                                  reason="Queue full / Latency check")
                    continue
                
                if self._trace_recorder:
                    self._trace_recorder.record("Worker", "Start Inference", 
                                              timestep=obs.get_timestep(),
                                              queue_size=self._action_queue.get_queue_size())
                
                # Run inference
                start_time = time.perf_counter()
                action_chunk = self._inference_fn(obs.images, obs.state)
                elapsed = time.perf_counter() - start_time
                
                # Update latency estimate
                self._latency_estimator.update(elapsed)
                
                # Convert to TimedActions
                timed_actions = self._time_action_chunk(
                    t_0=obs.get_timestamp(),
                    action_chunk=action_chunk,
                    i_0=obs.get_timestep()
                )
                
                # Add to action queue
                self._action_queue.add_action_chunk(timed_actions)
                
                # Signal that we've processed - next empty queue triggers must_go
                self._must_go_event.set()
                
                logger.debug(
                    f"queue_size={self._action_queue.get_queue_size()}"
                )
                
                if self._trace_recorder:
                    self._trace_recorder.record("Worker", "End Inference", 
                                              duration_ms=elapsed*1000,
                                              chunk_size=len(action_chunk),
                                              new_queue_size=self._action_queue.get_queue_size())
                
            except Exception as e:
                logger.error(f"Async inference error: {e}")
                import traceback
                traceback.print_exc()
    
    def _should_process_observation(self, obs: TimedObservation) -> bool:
        """
        Check if observation should be processed.
        
        LeRobot pattern:
        - must_go observations are always processed
        - Otherwise, check if we need new actions
        """
        if obs.must_go:
            return True
        
        # Check if action queue needs refilling
        return self._action_queue.should_request_new_chunk(self._latency_estimator)
    
    def _time_action_chunk(
        self,
        t_0: float,
        action_chunk: np.ndarray,
        i_0: int
    ) -> List[TimedAction]:
        """
        Convert action chunk to TimedAction list.
        
        LeRobot pattern (from policy_server.py _time_action_chunk):
        First action corresponds to t_0, rest are t_0 + i*dt
        """
        dt = self.config.environment_dt
        return [
            TimedAction(
                timestamp=t_0 + i * dt,
                timestep=i_0 + i,
                action=action_chunk[i]
            )
            for i in range(len(action_chunk))
        ]


# ==================== Base Inference Engine ====================

class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference policies with LeRobot-style async support.
    
    Key Features:
    - Timestamp-aligned action queue
    - Background inference thread
    - Latency-adaptive chunk threshold
    - Gripper velocity clamping
    """
    
    def __init__(self, smoothing_config: Optional[SmoothingConfig] = None):
        self.is_loaded = False
        self.model_type: str = ""
        self.required_cameras: List[str] = []
        self.state_dim: int = 0
        self.action_dim: int = 7
        self.chunk_size: int = 1
        self.n_action_steps: int = 1
        self.requested_device: Optional[str] = None
        self.actual_device: Optional[str] = None
        self.device_warning: str = ""
        
        # Config
        self.smoothing_config = smoothing_config or SmoothingConfig()
        
        # Components (initialized after model load)
        self._action_queue: Optional[TimestampedActionQueue] = None
        self._latency_estimator: Optional[LatencyEstimator] = None
        self._gripper_smoother: Optional[GripperSmoother] = None
        self._async_worker: Optional[AsyncInferenceWorker] = None
        self._trace_recorder: Optional[TraceRecorder] = None
        
        # Episode state
        self._episode_start_time: float = 0.0
        self._current_timestep: int = 0
        self._fallback_count: int = 0
    
    def _init_components(self):
        """Initialize all components after model is loaded."""
        if self.smoothing_config.enable_async_inference and self.n_action_steps <= 1:
            logger.warning(
                "%s model reports n_action_steps=%s; disabling async inference because a single-step policy "
                "cannot keep a 30Hz action queue filled.",
                self.model_type or "Inference",
                self.n_action_steps,
            )
            self.smoothing_config.enable_async_inference = False

        self._action_queue = TimestampedActionQueue(self.smoothing_config)
        self._action_queue.set_chunk_size(self.n_action_steps)
        
        self._latency_estimator = LatencyEstimator(
            alpha=self.smoothing_config.latency_ema_alpha,
            initial_value=0.1
        )
        
        self._gripper_smoother = GripperSmoother(
            self.smoothing_config,
            self.action_dim
        )
        
        if self.smoothing_config.enable_async_inference:
            self._async_worker = AsyncInferenceWorker(
                config=self.smoothing_config,
                inference_fn=self._predict_chunk,
                action_queue=self._action_queue,
                latency_estimator=self._latency_estimator,
                trace_recorder=self._trace_recorder,
            )
    
    @abstractmethod
    def load(self, checkpoint_dir: str) -> Tuple[bool, str]:
        """Load model from checkpoint directory."""
        pass
    
    @abstractmethod
    def _predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Predict action chunk from observation.
        
        Args:
            images: Dict of {camera_role: image array (H, W, 3)}
            state: Robot state array (state_dim,)
            
        Returns:
            Action chunk (n_action_steps, action_dim)
        """
        pass
    
    def reset(self):
        """Reset state for new episode."""
        # Stop async worker if running
        if self._async_worker is not None:
            self._async_worker.stop()
        
        # Reset all components
        if self._action_queue is not None:
            self._action_queue.reset()
        if self._gripper_smoother is not None:
            self._gripper_smoother.reset()
        
        # Reset episode state
        self._episode_start_time = time.time()
        self._current_timestep = 0
        self._fallback_count = 0
        
        logger.debug(f"{self.model_type} inference engine reset")
    
    def start_async_inference(self):
        """Start background inference thread."""
        if self._async_worker is not None:
            self._async_worker.start()
    
    def stop_async_inference(self):
        """Stop background inference thread."""
        if self._async_worker is not None:
            self._async_worker.stop()

    def predict_chunk(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Predict a raw action chunk without queue execution semantics.

        This is useful for offline validation and analysis tools that want the
        direct policy output instead of the single action selected by the
        control loop.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        return self._predict_chunk(images, state)

    def step(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """Public alias for one control-loop step."""
        return self.select_action(images, state)

    def select_action(self, images: Dict[str, np.ndarray], state: np.ndarray) -> np.ndarray:
        """
        Select action with LeRobot-style timestamp alignment.
        
        Flow:
        1. Create TimedObservation and submit to async worker
        2. Get action from queue (timestamp-aligned or fallback)
        3. Apply gripper smoothing
        4. Increment timestep
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        current_time = time.time()
        
        # Sync timestep with wall clock to handle loop lag
        # If loop is slower than control_fps, we need to skip timesteps to stay aligned
        elapsed = max(0.0, current_time - self._episode_start_time)
        self._current_timestep = int(elapsed / self.smoothing_config.environment_dt)
        
        # Create timed observation
        obs = TimedObservation(
            timestamp=current_time,
            timestep=self._current_timestep,
            images=images,
            state=state,
            must_go=False
        )
        
        # Submit to async worker (if enabled)
        if self._async_worker is not None and self._async_worker._running:
            self._async_worker.submit_observation(obs)
            
            # Get action from queue with timestamp alignment
            timed_action = self._action_queue.get_action_for_time(
                current_time, 
                self._episode_start_time
            )
            
            if timed_action is None and self._trace_recorder:
                self._trace_recorder.record("Engine", "Queue Empty", timestep=self._current_timestep)
        else:
            # Synchronous mode: run inference directly if queue empty
            timed_action = self._action_queue.get_next_action()
            
            if timed_action is None:
                # Run synchronous inference
                start_time = time.perf_counter()
                action_chunk = self._predict_chunk(images, state)
                elapsed = time.perf_counter() - start_time
                
                self._latency_estimator.update(elapsed)
                
                # Add to queue
                timed_actions = [
                    TimedAction(
                        timestamp=current_time + i * self.smoothing_config.environment_dt,
                        timestep=self._current_timestep + i,
                        action=action_chunk[i]
                    )
                    for i in range(len(action_chunk))
                ]
                self._action_queue.add_action_chunk(timed_actions)
                
                # Get first action
                timed_action = self._action_queue.get_next_action()
                
                logger.debug(f"Sync inference: {elapsed*1000:.1f}ms")
        
        # Handle empty queue (fallback)
        if timed_action is None:
            action = self._get_fallback_action(state)
            self._fallback_count += 1
            logger.debug(f"Using fallback action (count={self._fallback_count})")
        else:
            action = timed_action.get_action()
        
        # Apply gripper smoothing
        if self._gripper_smoother is not None:
            action = self._gripper_smoother.smooth(action)
        
        # Timestep is updated at start of method based on wall clock
        # self._current_timestep += 1
        
        return action
    
    def _get_fallback_action(self, state: np.ndarray) -> np.ndarray:
        """Get fallback action when queue is empty."""
        mode = self.smoothing_config.fallback_mode
        
        if mode == "repeat" and self._gripper_smoother is not None:
            last_action = self._gripper_smoother.get_last_action()
            if last_action is not None:
                return last_action
        
        # "hold" mode or no last action: return current state
        return state[:self.action_dim].copy() if len(state) >= self.action_dim else np.zeros(self.action_dim)
    
    # ==================== Status Methods ====================
    
    def get_queue_size(self) -> int:
        """Get current action queue size."""
        if self._action_queue is not None:
            return self._action_queue.get_queue_size()
        return 0
    
    def get_fallback_count(self) -> int:
        """Get count of fallback uses."""
        return self._fallback_count
    
    def get_latency_estimate(self) -> float:
        """Get current latency estimate in seconds."""
        if self._latency_estimator is not None:
            return self._latency_estimator.get_value()
        return 0.0
    
    def get_required_cameras(self) -> List[str]:
        """Return list of required camera roles."""
        return self.required_cameras
    
    def get_state_dim(self) -> int:
        """Return expected state dimension."""
        return self.state_dim

    def get_device_status(self) -> Dict[str, Optional[str]]:
        """Return requested/actual device metadata for observability."""
        return {
            "requested_device": self.requested_device,
            "actual_device": self.actual_device,
            "device_warning": self.device_warning,
        }
    
    def set_control_fps(self, fps: float):
        """Update control frequency."""
        self.smoothing_config.control_fps = fps
        if self._action_queue is not None:
            self._action_queue.config.control_fps = fps
    
    def set_smoothing_config(self, config: SmoothingConfig):
        """Update smoothing configuration."""
        self.smoothing_config = config
    
    @staticmethod
    def validate_checkpoint(checkpoint_dir: str) -> Tuple[bool, str]:
        """Validate that checkpoint directory contains required files."""
        path = Path(checkpoint_dir)
        
        if not path.exists():
            return False, f"Checkpoint目录不存在: {checkpoint_dir}"
        
        required_files = ["inference_config.yaml", "model.pth", "stats.json"]
        missing = []
        for f in required_files:
            if not (path / f).exists():
                missing.append(f)
        
        if missing:
            return False, f"缺少必需文件: {', '.join(missing)}"
        
        return True, ""
    
    @abstractmethod
    def unload(self):
        """Unload model and free memory."""
        pass
    def set_trace_recorder(self, recorder: TraceRecorder):
        """Set trace recorder for observability."""
        self._trace_recorder = recorder
        if self._async_worker is not None:
            self._async_worker._trace_recorder = recorder
