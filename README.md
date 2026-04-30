# Inference SDK

`inference-sdk` 是一个独立的 Python SDK，用于 ACT、SmolVLA 和 PI0 等 policy 模型推理。

它只关注一件事：

> 输入 observation，输出 action

硬件驱动、相机采集、Web API、会话编排等业务逻辑应放在上层业务应用中。

## 安装

推荐使用 `uv` 管理隔离环境：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ../SparkMind
```

可选依赖：

```bash
uv pip install -e .[act]
uv pip install -e .[examples]
uv pip install -e .[vla]
uv pip install -e .[all]
```

## 使用方式

推荐给业务应用使用的高层 SDK API：

```python
from inference_sdk import InferenceSDK, Observation

with InferenceSDK(device="cuda:0") as sdk:
    metadata = sdk.load_policy(
        algorithm_type="pi0",
        checkpoint_dir="/path/to/checkpoint",
        instruction="Pick up the object.",
    )

    observation = Observation(
        images={
            "head": head_bgr,
            "wrist": wrist_bgr,
        },
        state=robot_state,
    )

    action_chunk = sdk.predict_action_chunk("pi0", observation)
    print(action_chunk.shape)  # (metadata.n_action_steps, metadata.action_dim)
```

`images` 的 key 应使用 `metadata.required_cameras` 中返回的相机角色名；每张图像应是 BGR 格式的 numpy 数组，形状为 `(H, W, 3)`。

如果只需要执行一次推理，也可以使用一次性 API：

```python
from inference_sdk import predict_action_chunk

action_chunk = predict_action_chunk(
    algorithm_type="act",
    checkpoint_dir="/path/to/checkpoint",
    images=images,
    state=robot_state,
)
```

底层 engine API 仍然保留，适合需要直接控制加载、reset、step 等流程的场景：

```python
from inference_sdk import SmoothingConfig, create_engine

engine = create_engine(
    model_type="pi0",
    device="cuda:0",
    smoothing_config=SmoothingConfig(control_fps=30.0),
)

ok, error = engine.load("/path/to/checkpoint")
if not ok:
    raise RuntimeError(error)

engine.reset()
action = engine.select_action(images=images, state=state)
```

### 进程内异步推理

如果需要将动作执行和模型推理解耦，可以使用全局异步推理运行时。该方式不启动 gRPC server/client，而是在当前 Python 后端进程内维护观测队列、后台推理线程和动作队列。

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

result = runtime.step(images=images, state=robot_state)
robot.send_action(result.action)
```

完整控制循环模板见 `examples/async_runtime_loop.py`，设计方案见 `docs/async_inference_plan.md`。

如果需要运行示例，请先安装绘图和 Hugging Face Hub 相关依赖：

```bash
uv pip install -e ".[all,examples]"
```
