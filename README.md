# Inference SDK

`inference-sdk` is a standalone Python package for ACT, SmolVLA, and PI0 policy inference.

It focuses on one responsibility:

> input observation, output action

Hardware drivers, camera capture, web APIs, and session orchestration should stay in the business application.

## Install

Recommended: use `uv` for isolated environment management.

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -e .
```

Optional extras:

```bash
uv pip install -e .[act]
uv pip install -e .[examples]
uv pip install -e .[vla]
uv pip install -e .[all]
```

If you are using a local `SparkMind` checkout:

```bash
uv pip install -e ./SparkMind
```

`uv` is recommended, but not required. Plain `pip`, `venv`, or `conda` also work if you prefer them.

## Usage

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

## Runtime Environment

The SDK does not assume any host repository layout.

If you use a local SparkMind checkout instead of an installed package, set one of:

```bash
export INFERENCE_SDK_SPARKMIND_PATH=/absolute/path/to/SparkMind
export SPARKMIND_PATH=/absolute/path/to/SparkMind
```

If your tokenizer or VLM assets live in local model directories, you can point the SDK to them with:

```bash
export PI0_TOKENIZER_PATH=/absolute/path/to/tokenizer
export SMOLVLA_VLM_MODEL_PATH=/absolute/path/to/vlm
export INFERENCE_SDK_MODEL_ROOTS=/absolute/path/to/models
```

`INFERENCE_SDK_MODEL_ROOTS` accepts multiple paths separated by `:`.

## Examples

The repository includes an offline dataset validation example:

- `examples/validate_dataset_inference.py`
- `examples/validate_dataset_inference.md`

Install the extra plotting / Hub dependencies first if you want to run it:

```bash
uv pip install -e ".[all,examples]"
```
