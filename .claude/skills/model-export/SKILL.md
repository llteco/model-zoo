---
name: "model-export"
description: "Use when exporting registered models to ONNX, registering new models for export, understanding export commands/options, or troubleshooting ONNX export in this model-zoo project."
---

# Model Export Guide

## Overview

This skill covers how to export registered PyTorch models to ONNX format in this model-zoo repository.

## Registry System

Models are registered using a decorator-based registry in `src/zoo/export/__init__.py`. The `EXPORT` registry auto-imports modules from `src/zoo/export/{vision,vlm,asr,tts}/`.

### Registration Pattern

```python
from zoo.export import EXPORT, export_post_process

@EXPORT.register("model_name")
@export_post_process(["fold_constant", "eliminate_dead_nodes"])  # optional
class MyModel(torch.nn.Module):
    hier = [...]           # optional: sub-modules for hierarchical export
    input_names = (...)    # optional: ONNX input names
    output_names = (...)   # optional: ONNX output names

    def __init__(self, model_name="...", **kwargs):
        super().__init__()
        # Load and configure model

    def forward(self, *inputs):
        # Forward pass

    @property
    def default_inputs(self):
        # Default inputs for export
        return {...}
```

## Export Command

### Basic Command

```bash
uv -m zoo.export <module_name> [input_shapes] [options] [init_args]
```

### Available Options

| Option | Description |
|--------|-------------|
| `module` | Registered model name (required) |
| `input_shapes` | Input tensor shapes, e.g., `1,3,224,224` |
| `--dynamo` | Use Torch Dynamo for export |
| `--opset-version, -v` | ONNX opset version (default: 23) |
| `--device` | Device: `cpu` or `cuda` (default: cpu) |
| `--dtype` | Data type: `float16`, `bfloat16`, `float32` |
| `--external-directory` | Directory for external tensor data |
| `--export-with-hier, -hier` | Export with hierarchical structure |
| `--no-post-process` | Skip post-processing |
| `--man, -m` | Print model signature info |

### Example Commands

```bash
# Export a vision model
uv -m zoo.export qwen3.5vision

# Export text model with custom seq_len
uv -m zoo.export qwen3.5text --seq_len 32

# Export with hierarchical structure
uv -m zoo.export qwen3.5text -hier

# Check model signature before export
uv -m zoo.export -m qwen3.5vision

# Export with custom dtype
uv -m zoo.export resnet50 --dtype float16 --device cuda
```

## Creating a New Export Module

### Step 1: Choose Category

Create file in appropriate subdirectory:
- `src/zoo/export/vision/` - Vision models
- `src/zoo/export/vlm/` - Vision-language models
- `src/zoo/export/asr/` - Speech recognition
- `src/zoo/export/tts/` - Text-to-speech

### Step 2: Define Export Class

Key components:
1. `@EXPORT.register("name")` decorator
2. `@export_post_process([...])` for ONNX optimization
3. `default_inputs` property for test inputs
4. `forward()` method matching expected input/output

### Step 3: Run Export

```bash
uv -m zoo.export <your_model_name>
```

Output: `<model_name>.onnx` in current directory.

## Post-Processing Options

Available post-processors in `export_post_process()`:
- `fold_constant` - Fold constant nodes
- `eliminate_dead_nodes` - Remove unused nodes
- `eliminate_nop_slice` - Remove no-op slice operations
- `trt_vit_attention_replace` - TensorRT ViT attention optimization
- `remove_unused_functions` - Clean up function definitions

## Troubleshooting

### Common Issues

**Export fails with shape mismatch**: Check `default_inputs` matches `forward()` signature. Use `-m` flag to inspect expected inputs.

**Large ONNX file**: Use `--external-directory` to separate tensor data, or enable `fold_constant`.

**Dynamic shapes**: Define shapes in `input_shapes` argument or use `--dynamo` for dynamic export.

**Custom model args**: Pass init arguments after options, e.g., `uv -m zoo.export model --arg1 value1`.