---
name: "benchmark"
description: "Use when running benchmarks, choosing bench command arguments, or interpreting latency and repeat results."
---

# Benchmark Runner

## Overview

Use this skill to run benchmark commands and interpret latency results for models in this repository.

## How to Run Benchmark Commands

### 1. Basic Benchmark Command

```bash
# Example benchmark command
uv run bench torch_conv2d 1,256,512,512:float16 --channels 256 --layers 10 --iters 10 --warmup 5 --repeat 3 --reduce median
```

### 2. Available Options

- `model name`: Specify the model name to benchmark, e.g., `torch_conv2d`
- `inputs spec`: Specify the input shapes and types, if there are multiple inputs, separate them with space, e.g., `1,256,512,512:float16 1,256,512,512:float16`
- `--warmup`: Number of warmup iterations, default is 5
- `--iters`: Number of benchmark iterations, default is 10
- `--repeat`: Number of repeat times, default is 1
- `--reduce`: A string of reduction methods, default is `None`, choices are `[mean, median, min, max]`
- `--compile`: Whether to torch.compile the model, default is `False`
- `--half`: Whether to use half precision, default is `False`
- `--device`: Specify device (cpu/cuda/xpu), default is `cuda`
- `--other-args`: arguments from model's `__init__` method, e.g., `--channels 256 --layers 10`
- `-m`: query model's `__init__` method arguments, e.g., `uv run bench -m torch_conv2d`

## Interpreting Results

The result is either a single number or a list of numbers, depending on the `--reduce` option.
- If `--reduce` is `None`, the result is a list of numbers, each number is average latency of each repeat.
- If `--reduce` is `mean`, `median`, `min`, or `max`, the result is a single number, which is the mean, median, minimum, or maximum of the latency list, respectively.

## Troubleshooting

### Common Issues

- **Precision mismatch**: If input spec uses `float16`, make sure --half is set.
- **Compile error**: Some models can't be compiled by torch.compile.
- **Arguments mismatch**: Input shapes do not match the model's `__init__` method arguments.
