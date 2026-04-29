# Inference SDK 异步推理执行方案

## 1. 背景与目标

本方案参考 LeRobot 的异步推理思想，但不采用 `PolicyServer` / `RobotClient` 的 gRPC 拆分方式，而是在当前 `Inference SDK` 进程内实现一个可被后端和前端间接调用的全局异步推理运行时。

核心目标：

- 将“观测采集 / 动作执行”和“策略模型推理”解耦。
- 使用观测队列始终保留最新观测，避免模型处理过期帧。
- 使用动作队列缓存未来动作，减少模型推理期间的机器人 idle frames。
- 提供全局运行时 API，方便后端控制循环、HTTP 接口、WebSocket 状态推送等上层业务调用。
- 不把网络通信、机器人硬件控制、相机采集逻辑放进 SDK，保持 SDK 只关注推理调度。

## 2. 设计原则

### 2.1 不使用 server/client

LeRobot 的异步推理由 gRPC 服务端和机器人客户端组成，适合跨进程或跨机器部署。本项目暂不需要这层网络拆分，因此采用进程内异步线程模型：

```text
业务后端 / 控制循环
        │
        ▼
全局 AsyncInferenceRuntime
        │
        ├── ObservationQueue：保存最新观测
        ├── Async Worker：后台模型推理
        └── ActionQueue：保存可执行动作
```

### 2.2 SDK 只提供推理运行时

SDK 不直接负责：

- 启动 Web 服务。
- 连接真实机器人。
- 采集相机图像。
- 解析前端上传的数据格式。
- 调用真实硬件执行 action。

这些能力应由上层业务系统实现。SDK 提供稳定、线程安全的 Python API，供上层调用。

### 2.3 全局运行时是进程内全局

“全局”指同一个 Python 后端进程中的 singleton，例如：

```python
from inference_sdk import get_global_async_runtime

runtime = get_global_async_runtime()
```

如果前端是浏览器或 Electron UI，不能直接访问 Python 内存对象，需要通过后端 HTTP / WebSocket 接口访问该全局 runtime。

## 3. 当前项目基础

当前项目已经具备实现异步推理的关键基础：

- 高层推理入口：`InferenceSDK`。
- 单次动作 chunk 推理：`InferenceSDK.predict_action_chunk(...)`。
- 动作队列：`TimestampedActionQueue`。
- 观测队列：`ObservationQueue`。
- 动作时间戳数据结构：`TimedAction`。
- 观测时间戳数据结构：`TimedObservation`。
- 平滑和聚合配置：`SmoothingConfig`。
- 聚合函数：`latest_only`、`weighted_average`、`average`、`conservative`。

因此建议不要重写底层队列，而是在现有能力上封装一个更适合前后端调用的运行时模块。

## 4. 目标模块

新增模块：

```text
inference_sdk/async_runtime.py
```

该模块负责：

- 加载和持有 `InferenceSDK` 实例。
- 管理当前策略模型。
- 管理后台推理线程。
- 管理观测队列和动作队列。
- 提供 `submit_observation()`、`get_action()`、`step()` 等控制循环 API。
- 提供 `get_status()` 给后端或前端监控使用。
- 提供全局 singleton，避免业务层重复加载模型。

## 5. 核心类设计

### 5.1 `AsyncInferenceConfig`

用于描述异步调度参数，可以复用或包装现有 `SmoothingConfig`。

建议字段：

```python
@dataclass
class AsyncInferenceConfig:
    control_fps: float = 30.0
    chunk_size_threshold: float = 0.5
    aggregate_fn_name: str = "weighted_average"
    obs_queue_maxsize: int = 1
    fallback_mode: str = "repeat"
    latency_ema_alpha: float = 0.2
    latency_safety_margin: float = 1.5
    enable_gripper_clamping: bool = True
    gripper_max_velocity: float = 200.0
```

字段说明：

- `control_fps`：控制循环频率。
- `chunk_size_threshold`：动作队列剩余比例低于该阈值时提交新观测。
- `aggregate_fn_name`：重叠 timestep 的新旧动作合并策略。
- `obs_queue_maxsize`：观测队列大小，默认 1，只保留最新观测。
- `fallback_mode`：动作队列为空时的回退策略。
- `latency_ema_alpha`：推理耗时 EMA 更新系数。
- `latency_safety_margin`：根据推理延迟预留动作数量的安全系数。
- `enable_gripper_clamping`：是否启用夹爪速度限制。
- `gripper_max_velocity`：夹爪单步最大变化量。

### 5.2 `AsyncRuntimeStatus`

用于前后端查询 runtime 状态。

建议字段：

```python
@dataclass(frozen=True)
class AsyncRuntimeStatus:
    loaded: bool
    running: bool
    model_type: str | None
    checkpoint_dir: str | None
    queue_size: int
    fill_ratio: float
    latency_estimate: float
    fallback_count: int
    current_timestep: int
    last_error: str | None
    worker_alive: bool
```

用途：

- 后端日志记录。
- 前端状态面板展示。
- 判断动作队列是否经常耗尽。
- 判断后台推理线程是否异常退出。

### 5.3 `AsyncStepResult`

用于 `step(...)` 返回单帧控制结果。

建议字段：

```python
@dataclass(frozen=True)
class AsyncStepResult:
    action: np.ndarray
    source: str
    timestep: int
    submitted_observation: bool
    queue_size: int
    latency_estimate: float
```

`source` 可取值：

- `queue`：动作来自动作队列。
- `fallback_repeat`：动作队列为空，重复上一帧动作。
- `fallback_hold`：动作队列为空，使用当前状态或零动作。
- `sync_warmup`：可选，启动初期同步推理一帧用于填充队列。

### 5.4 `AsyncInferenceRuntime`

核心运行时类。

建议方法：

```python
class AsyncInferenceRuntime:
    def load_policy(...): ...
    def start(self): ...
    def stop(self): ...
    def reset(self): ...
    def close(self): ...

    def submit_observation(...): ...
    def get_action(...): ...
    def step(...): ...
    def predict_action_chunk(...): ...
    def get_status(self): ...
```

方法职责：

- `load_policy(...)`：加载策略模型，并初始化队列、延迟估计器和平滑器。
- `start()`：启动后台 worker。
- `stop()`：停止后台 worker，但不卸载模型。
- `reset()`：清空队列，重置 timestep、fallback 计数和平滑状态。
- `close()`：停止 worker 并卸载模型。
- `submit_observation(...)`：提交最新观测到观测队列。
- `get_action(...)`：从动作队列取当前 timestep 对应动作。
- `step(...)`：控制循环推荐入口，内部完成提交观测和取动作。
- `predict_action_chunk(...)`：保留同步动作 chunk 推理入口，方便调试或离线验证。
- `get_status()`：返回运行状态。

## 6. 运行流程

### 6.1 初始化流程

```text
后端启动
  │
  ├── get_global_async_runtime()
  │
  ├── runtime.load_policy(...)
  │
  ├── runtime.start()
  │
  └── 等待控制循环调用 step(...)
```

初始化时需要完成：

1. 创建 `InferenceSDK`。
2. 加载指定策略模型。
3. 获取模型 metadata，例如 `n_action_steps`、`action_dim`、`required_cameras`。
4. 根据 metadata 设置动作队列 chunk size。
5. 初始化 `ObservationQueue(maxsize=1)`。
6. 初始化 `TimestampedActionQueue`。
7. 初始化后台 worker，但不立即执行推理。

### 6.2 控制循环流程

推荐上层控制循环每个控制周期调用一次：

```python
result = runtime.step(images=images, state=state)
robot.send_action(result.action)
```

`step(...)` 内部流程：

```text
采集当前时间和 timestep
  │
  ├── 判断是否需要提交新观测
  │     ├── 队列为空：must_go=True
  │     └── 队列低于阈值：submit_observation(...)
  │
  ├── 从动作队列取当前 timestep 的动作
  │     ├── 命中：返回 queue action
  │     └── 未命中：返回 fallback action
  │
  └── 返回 AsyncStepResult
```

### 6.3 后台 worker 流程

后台 worker 持续执行：

```text
等待 ObservationQueue 中的新观测
  │
  ├── 判断是否应处理
  │     ├── must_go=True：必须处理
  │     └── 动作队列未低于阈值：跳过
  │
  ├── 调用 sdk.predict_action_chunk(...)
  │
  ├── 更新推理延迟估计
  │
  ├── 将 action chunk 转为 TimedAction 列表
  │
  └── 写入 TimestampedActionQueue
```

## 7. 队列策略

### 7.1 观测队列

观测队列使用 `maxsize=1`。

原因：

- 控制循环和相机采集通常比模型推理更快。
- 如果缓存多个观测，后台模型可能一直处理过期帧。
- 只保留最新观测，可以提高动作计划对当前状态的适应性。

写入策略：

```text
如果队列未满：直接写入
如果队列已满：丢弃旧观测，写入新观测
```

### 7.2 动作队列

动作队列按 `timestep` 管理，而不是简单 FIFO。

动作 chunk 写入规则：

1. 丢弃 `timestep <= latest_executed_timestep` 的过期动作。
2. 如果 timestep 不存在，直接加入队列。
3. 如果 timestep 已存在，使用 `aggregate_fn_name` 对新旧动作聚合。

动作读取规则：

1. 根据 `episode_start_time` 和 `control_fps` 计算当前应执行的 timestep。
2. 优先取当前 timestep 对应动作。
3. 如果当前 timestep 不存在，取最近的未来动作。
4. 丢弃已经过期的旧动作。

## 8. 关键参数建议

### 8.1 `control_fps`

建议默认：`30.0`。

调参建议：

- 如果动作队列经常为空，降低 `control_fps`。
- 如果机器人控制需要更高响应，可提高 `control_fps`，但模型推理和动作 chunk 必须跟得上。

### 8.2 `chunk_size_threshold`

建议默认：`0.5`。

含义：

```text
action_queue_size / action_chunk_size <= chunk_size_threshold
```

调参建议：

- `0.3`：更少提交观测，推理压力低，但适应性弱。
- `0.5`：推荐起点。
- `0.8`：更频繁提交观测，适应性强，但推理压力高。

### 8.3 `aggregate_fn_name`

建议默认：`weighted_average`。

可选策略：

- `latest_only`：完全采用新动作，响应快但可能抖动。
- `weighted_average`：`0.3 * old + 0.7 * new`，推荐默认。
- `average`：新旧动作平均。
- `conservative`：`0.7 * old + 0.3 * new`，更平滑但响应慢。

### 8.4 `fallback_mode`

建议默认：`repeat`。

可选策略：

- `repeat`：队列为空时重复上一帧动作。
- `hold`：队列为空时保持当前状态或返回零动作。

真实机器人场景中，`repeat` 通常更平滑；安全要求更高时，可由上层业务把 fallback 转换成停机或 hold pose。

## 9. 全局 API 设计

### 9.1 获取全局 runtime

```python
from inference_sdk import get_global_async_runtime

runtime = get_global_async_runtime()
```

### 9.2 加载模型

```python
metadata = runtime.load_policy(
    algorithm_type="pi0",
    checkpoint_dir="/path/to/checkpoint",
    device="cuda:0",
    instruction="Pick up the object.",
    config=AsyncInferenceConfig(
        control_fps=30.0,
        chunk_size_threshold=0.5,
        aggregate_fn_name="weighted_average",
    ),
)
```

### 9.3 启动异步推理

```python
runtime.start()
```

### 9.4 控制循环调用

```python
while running:
    images = camera.read_images()
    state = robot.read_state()

    result = runtime.step(images=images, state=state)
    robot.send_action(result.action)

    sleep_until_next_control_tick()
```

### 9.5 查询状态

```python
status = runtime.get_status()

print(status.queue_size)
print(status.latency_estimate)
print(status.fallback_count)
```

### 9.6 停止和释放

```python
runtime.stop()
runtime.close()
```

## 10. 后端接口包装建议

如果上层后端使用 FastAPI，可将全局 runtime 包装为 HTTP 接口。

### 10.1 推荐接口

```text
POST /inference/load
POST /inference/start
POST /inference/stop
POST /inference/reset
POST /inference/step
POST /inference/observation
GET  /inference/action
GET  /inference/status
```

接口职责：

- `/inference/load`：加载模型。
- `/inference/start`：启动后台推理线程。
- `/inference/stop`：停止后台推理线程。
- `/inference/reset`：重置队列和 episode 状态。
- `/inference/step`：提交观测并返回一个动作，适合后端控制循环调用。
- `/inference/observation`：只提交观测，不立即取动作。
- `/inference/action`：只获取当前可执行动作。
- `/inference/status`：返回队列、延迟、fallback、线程状态等信息。

### 10.2 前端调用建议

前端通常不直接上传高频大图像给 SDK，推荐两种方式：

1. 前端只负责显示状态，后端本地采集相机和机器人状态。
2. 如果必须由前端上传图像，应通过后端进行图像解码、校验、限流和格式转换。

前端重点展示：

- 当前模型类型。
- 是否 running。
- 动作队列大小。
- 推理延迟估计。
- fallback 次数。
- 最近错误信息。

## 11. 线程安全策略

运行时需要保证以下操作线程安全：

- `load_policy()` 与 `close()` 互斥。
- `start()` 与 `stop()` 互斥。
- `submit_observation()` 可被控制循环频繁调用。
- `get_action()` 可被控制循环频繁调用。
- 后台 worker 写动作队列时，不影响主线程读动作队列。

建议实现：

- runtime 内部使用 `threading.RLock` 管理生命周期。
- 观测队列和动作队列保持各自内部锁。
- `last_error` 使用锁保护或只在 runtime 锁内更新。
- `close()` 必须先停止 worker，再卸载模型。

## 12. 错误处理策略

### 12.1 模型未加载

如果调用 `start()`、`step()`、`submit_observation()` 或 `get_action()` 时模型未加载，应抛出明确错误：

```text
RuntimeError: Async inference runtime has no loaded policy. Call load_policy() first.
```

### 12.2 后台推理异常

后台 worker 不应因单次推理异常直接杀死整个后端进程。

建议行为：

1. 捕获异常。
2. 记录 `last_error`。
3. 增加错误计数。
4. 继续等待下一帧观测。
5. 如果连续错误超过阈值，可将 `running=False` 并等待人工 reset。

### 12.3 动作队列为空

动作队列为空时：

1. 返回 fallback action。
2. 增加 `fallback_count`。
3. 下一次观测标记为 `must_go=True`。
4. 状态接口暴露 fallback 次数，用于调参。

## 13. 实施步骤

### 阶段一：运行时模块

新增 `inference_sdk/async_runtime.py`：

- 定义 `AsyncInferenceConfig`。
- 定义 `AsyncRuntimeStatus`。
- 定义 `AsyncStepResult`。
- 定义 `AsyncInferenceRuntime`。
- 定义 `get_global_async_runtime()`。
- 定义便捷函数，例如 `load_async_policy()`、`async_step()`、`get_async_status()`。

### 阶段二：包导出

更新 `inference_sdk/__init__.py`：

- 导出 `AsyncInferenceConfig`。
- 导出 `AsyncRuntimeStatus`。
- 导出 `AsyncStepResult`。
- 导出 `AsyncInferenceRuntime`。
- 导出 `get_global_async_runtime()`。

### 阶段三：测试验证

建议增加最小测试或示例，使用 mock 推理函数验证：

- 观测队列满时丢弃旧观测。
- 后台 worker 能将 action chunk 写入动作队列。
- `step()` 能持续返回 action。
- 动作队列为空时 fallback 生效。
- 重叠 chunk 能按指定 aggregate 函数聚合。
- `stop()` 和 `close()` 能正常释放线程。

### 阶段四：文档和示例

更新 `README.md` 或新增示例文档：

- 给出后端控制循环示例。
- 给出 FastAPI 包装示例。
- 给出调参建议。
- 说明全局 runtime 是进程内全局，不是跨进程共享。

### 阶段五：业务接入

由上层业务系统完成：

- 相机图像采集。
- 机器人状态读取。
- action 执行。
- HTTP / WebSocket API。
- 前端状态展示。
- 异常停机和安全策略。

## 14. 建议目录结构

```text
inference_sdk/
  __init__.py
  api.py
  base.py
  async_runtime.py        # 新增
  factory.py
  monitor.py
  policy/

examples/
  async_runtime_loop.py   # 可选新增

docs/
  async_inference_plan.md # 当前文档
```

## 15. 最小使用示例

```python
from inference_sdk import AsyncInferenceConfig, get_global_async_runtime


runtime = get_global_async_runtime()

runtime.load_policy(
    algorithm_type="act",
    checkpoint_dir="/path/to/checkpoint",
    device="cuda:0",
    config=AsyncInferenceConfig(
        control_fps=30.0,
        chunk_size_threshold=0.5,
        aggregate_fn_name="weighted_average",
    ),
)

runtime.start()

try:
    while True:
        images = read_camera_images()
        state = read_robot_state()

        result = runtime.step(images=images, state=state)
        send_robot_action(result.action)

        wait_next_tick()
finally:
    runtime.stop()
```

## 16. FastAPI 包装示例

```python
from fastapi import FastAPI
from inference_sdk import get_global_async_runtime


app = FastAPI()
runtime = get_global_async_runtime()


@app.post("/inference/start")
def start_inference():
    runtime.start()
    return {"ok": True}


@app.post("/inference/stop")
def stop_inference():
    runtime.stop()
    return {"ok": True}


@app.get("/inference/status")
def get_status():
    status = runtime.get_status()
    return status.__dict__
```

实际生产环境中，`/inference/step` 如果需要传输图像，应增加：

- 图像大小限制。
- 图像格式校验。
- 请求频率限制。
- base64 / multipart 解码。
- BGR / RGB 转换约定。
- 异常动作保护。

## 17. 风险与注意事项

### 17.1 Python 进程内全局不等于跨进程全局

如果后端使用多 worker 部署，例如 `uvicorn --workers 4`，每个 worker 都会有自己的全局 runtime 和模型副本。

建议：

- 真实机器人控制服务使用单 worker。
- 或者单独启动一个机器人控制进程，Web 后端只做命令转发。

### 17.2 GPU 模型不适合频繁加载卸载

模型加载应在后端启动或任务切换时执行，不应每次请求都加载。

### 17.3 前端不应高频直接驱动真实机器人

真实机器人控制循环应在后端本地稳定运行，前端负责发起任务、展示状态和人工干预。

### 17.4 fallback 不是安全停机策略

fallback 只用于减少短暂动作断流。真正的安全停机、急停、限位保护应由机器人控制层实现。

## 18. 实现注意事项

### 18.1 避免双层异步

`AsyncInferenceRuntime` 是当前唯一的异步调度层；底层 `BaseInferenceEngine` 只负责同步动作 chunk 推理和直接 `step()` 队列执行。这样避免了两层异步调度：

```text
AsyncInferenceRuntime worker  ✅
BaseInferenceEngine worker    ❌ 已移除
```

双层异步会带来以下问题：

- timestep 对齐关系变复杂。
- 动作队列可能被两套逻辑同时管理。
- fallback 计数和状态监控不准确。
- 推理触发阈值可能重复判断。

因此，`AsyncInferenceRuntime` 加载底层 `InferenceSDK` 时仍会传入同步配置：

```python
from inference_sdk import InferenceSDK, SmoothingConfig


sdk = InferenceSDK(
    device=device,
    smoothing_config=SmoothingConfig(enable_async_inference=False),
)
```

之后 runtime worker 调用：

```python
sdk.predict_action_chunk(algorithm_type, images=images, state=state)
```

这样可以保证：

- runtime 统一管理观测队列。
- runtime 统一管理动作队列。
- runtime 统一维护 timestep、fallback、latency 和状态指标。
- 底层 policy 只做纯推理，不参与控制循环调度。

### 18.2 使用 `time.monotonic()`

动作 timestep、episode elapsed time、推理 latency 都建议使用 `time.monotonic()`，不要使用 `time.time()`。

原因：

- `time.time()` 是系统墙上时间，可能被 NTP 或用户手动调整。
- `time.monotonic()` 单调递增，更适合控制循环和耗时统计。

建议约定：

```python
episode_start_time = time.monotonic()
current_time = time.monotonic()
elapsed = current_time - episode_start_time
current_timestep = int(elapsed / environment_dt)
```

### 18.3 观测数据拷贝策略

观测进入队列前，需要明确数据所有权。

上层业务可能会复用 numpy buffer，例如相机线程持续写入同一块图像数组。如果 runtime 只保存引用，后台 worker 可能读到已经被下一帧覆盖的数据。

建议第一版采用安全策略：

- `state` 入队时使用 `np.asarray(state, dtype=np.float32).copy()`。
- `images` 入队时对每张图像执行 `np.asarray(image).copy()`。
- 文档中明确 `submit_observation()` 会复制输入数据。

如果后续需要极致性能，可增加配置：

```python
copy_observation: bool = True
```

当 `copy_observation=False` 时，由上层保证传入 buffer 在推理完成前不会被修改。

### 18.4 动作数据校验

后台 worker 拿到 `action_chunk` 后，不应直接写入动作队列，建议先做校验：

- `action_chunk` 必须是二维数组。
- `action_chunk.shape[1]` 必须等于 `action_dim`。
- 不允许包含 `NaN` 或 `Inf`。
- 如果配置了动作范围，应执行 clip 或拒绝该 chunk。

建议错误行为：

- shape 错误：记录 `last_error`，丢弃该 chunk。
- `NaN` / `Inf`：记录 `last_error`，丢弃该 chunk。
- 超出安全范围：按配置选择 clip 或丢弃。

### 18.5 单 policy 边界

第一版建议明确限制：

```text
一个 AsyncInferenceRuntime 同一时间只管理一个 policy。
```

如果需要切换模型，应执行：

```text
stop() -> close() 或 unload -> load_policy(new_policy) -> start()
```

不建议在 worker 运行时直接替换模型，因为这会带来：

- 队列中的旧动作来自旧模型。
- 新观测可能被旧 worker 处理。
- metadata，例如 `action_dim` 和 `n_action_steps` 可能变化。

未来如果需要多机器人或多模型，可在上层维护多个 runtime 实例，并为每个实例分配 `runtime_id` 或 `robot_id`。

## 19. 状态机设计

为了让后端和前端更容易判断当前 runtime 是否可用，建议定义明确状态机。

### 19.1 状态定义

```text
UNLOADED：未加载模型
LOADED：模型已加载，worker 未启动
RUNNING：worker 正在运行
STOPPED：worker 已停止，模型仍在内存中
CLOSED：runtime 已释放
ERROR：后台推理或生命周期操作发生不可恢复错误
```

### 19.2 状态流转

```text
UNLOADED
  │ load_policy()
  ▼
LOADED
  │ start()
  ▼
RUNNING
  │ stop()
  ▼
STOPPED
  │ start()
  ▼
RUNNING

LOADED / STOPPED / RUNNING
  │ close()
  ▼
CLOSED

RUNNING
  │ 连续推理失败或 worker 异常退出
  ▼
ERROR
```

### 19.3 方法调用约束

建议约束如下：

| 方法 | 允许状态 | 行为 |
| --- | --- | --- |
| `load_policy()` | `UNLOADED`、`LOADED`、`STOPPED` | 加载或重载模型；如果已运行，应要求先 `stop()` |
| `start()` | `LOADED`、`STOPPED` | 启动 worker |
| `stop()` | `RUNNING` | 停止 worker，保留模型 |
| `reset()` | `LOADED`、`RUNNING`、`STOPPED`、`ERROR` | 清空队列和计数，必要时从错误状态恢复到 `STOPPED` |
| `step()` | `RUNNING` | 提交观测并返回动作 |
| `submit_observation()` | `RUNNING` | 写入最新观测 |
| `get_action()` | `RUNNING` | 返回队列动作或 fallback |
| `close()` | 任意非 `CLOSED` 状态 | 停止 worker 并释放模型 |
| `get_status()` | 任意状态 | 返回当前状态 |

如果状态不允许调用某个方法，应抛出明确异常，避免静默失败。

## 20. 首帧预热与安全 fallback

### 20.1 首帧问题

异步推理启动后，动作队列初始为空。此时如果控制循环立即调用 `step()`，通常会出现短暂 fallback。

这不是 bug，而是异步流水线的自然现象：

```text
第 0 帧提交观测
后台 worker 开始推理
主线程立即需要 action
动作队列尚未填充
返回 fallback
```

### 20.2 可选预热策略

可以提供三种策略：

#### 策略一：允许 fallback

启动最快，实现最简单。

```text
start() 后直接进入控制循环，前几帧可能 fallback。
```

适合：

- 仿真环境。
- 对首帧动作不敏感的任务。
- fallback action 足够安全的场景。

#### 策略二：同步 warmup 一次

启动时同步推理一次，把第一个 action chunk 放入动作队列。

```python
runtime.warmup(images=images, state=state)
runtime.start()
```

适合：

- 真实机器人。
- 不希望启动时出现 idle 或 fallback 的场景。

缺点：

- `warmup()` 会阻塞一次完整模型推理耗时。

#### 策略三：异步 warmup 等待队列填充

先提交一帧观测，然后等待动作队列达到最小长度。

```python
runtime.submit_observation(images=images, state=state, must_go=True)
runtime.wait_until_ready(min_queue_size=1, timeout=5.0)
```

适合：

- 希望保持异步 worker 路径一致。
- 可以接受启动阶段等待的场景。

### 20.3 safe action 策略

fallback 不等于安全停机。建议 runtime 提供 `safe_action` 配置或回调：

```python
safe_action: np.ndarray | None = None
safe_action_fn: Callable[[np.ndarray], np.ndarray] | None = None
```

优先级建议：

1. 如果提供 `safe_action_fn`，使用它根据当前 state 生成安全动作。
2. 如果提供 `safe_action`，直接返回该动作。
3. 如果 `fallback_mode="repeat"` 且存在上一帧动作，重复上一帧动作。
4. 如果 `fallback_mode="hold"`，根据当前 state 生成 hold action。
5. 最后返回零动作。

真实机器人中，急停、限位保护和碰撞保护仍应由机器人控制层负责。

## 21. 监控指标与调试接口

### 21.1 建议新增指标

除基础状态外，建议 `AsyncRuntimeStatus` 增加以下监控字段：

```python
inference_count: int
processed_observation_count: int
dropped_observation_count: int
skipped_observation_count: int
action_chunk_count: int
queue_empty_count: int
fallback_count: int
error_count: int
consecutive_error_count: int
last_inference_ms: float | None
max_inference_ms: float | None
last_observation_timestep: int | None
last_action_timestep: int | None
```

字段用途：

- `inference_count`：累计推理次数。
- `processed_observation_count`：实际被模型处理的观测数。
- `dropped_observation_count`：观测队列满时丢弃的观测数。
- `skipped_observation_count`：因动作队列足够而跳过的观测数。
- `action_chunk_count`：写入动作队列的 chunk 数。
- `queue_empty_count`：动作队列为空次数。
- `fallback_count`：fallback 动作使用次数。
- `error_count`：累计错误次数。
- `consecutive_error_count`：连续错误次数。
- `last_inference_ms`：最近一次推理耗时。
- `max_inference_ms`：历史最大推理耗时。
- `last_observation_timestep`：最近提交观测 timestep。
- `last_action_timestep`：最近执行动作 timestep。

### 21.2 调试接口

建议提供以下调试方法：

```python
runtime.get_status()
runtime.get_queue_snapshot(limit=20)
runtime.get_trace_events(limit=100)
runtime.clear_metrics()
```

说明：

- `get_status()`：用于前端状态面板和健康检查。
- `get_queue_snapshot()`：返回动作队列中未来 timestep 的简要信息，不返回大数组或只返回 shape。
- `get_trace_events()`：返回最近事件，例如提交观测、开始推理、结束推理、队列为空、fallback。
- `clear_metrics()`：清空计数器，方便一次任务开始时重新统计。

### 21.3 后端部署建议

如果使用 FastAPI 或类似后端框架，真实机器人控制循环不建议由高频 HTTP 请求直接驱动。

推荐部署方式：

```text
后端进程
  ├── Robot Control Thread：固定频率采集观测、调用 runtime.step()、执行动作
  ├── AsyncInferenceRuntime Worker：后台模型推理
  └── Web API / WebSocket：接收任务命令、返回状态、推送监控数据
```

这样可以避免：

- HTTP 请求抖动影响控制频率。
- 前端刷新或网络延迟导致机器人动作断流。
- 多个前端请求同时驱动同一个机器人。

如果只是离线调试或仿真验证，可以临时用 `/inference/step` HTTP 接口驱动，但真实机器人建议使用后端本地控制线程。

## 22. 总结

本方案将 LeRobot 的异步推理思想改造成进程内运行时：

```text
最新观测队列 + 后台模型推理 + timestamp 动作队列 + 全局 runtime API
```

它保留了 LeRobot 异步推理的关键优点：

- 推理和动作执行并行。
- 观测队列只处理最新帧。
- 动作队列提前缓存未来动作。
- 重叠动作 chunk 可聚合修正。
- 动作队列为空时有 fallback 和 must-go 机制。

同时去掉了当前项目不需要的 gRPC server/client 部署复杂度，使后端可以直接通过 Python API 调用，前端则通过后端接口间接访问全局异步推理能力。
