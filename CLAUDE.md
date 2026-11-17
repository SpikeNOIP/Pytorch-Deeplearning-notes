# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A flat collection of numbered PyTorch tutorial scripts (Chinese-language) demonstrating core tensor operations. Each `.py` file is a standalone demo — there is no shared package, no build system, and no test suite.

## Environment

- Python: Anaconda at `D:\Anaconda` (Python 3.13.9)
- PyTorch: `torch 2.11.0+cpu` (CPU-only build — no CUDA)
- NumPy: `2.3.5`

## Running scripts

Scripts are run directly by filename (they contain Chinese characters; quote them):

```bash
python "01张量基本创建方法.py"
python "02 创建全1全0张量.py"
```

Each script is guarded by `if __name__ == "__main__":`, so it is safe to import without side effects.

## File conventions

- Files are numbered in teaching order (`01`, `02`, …) followed by a Chinese description of the topic.
- Each file defines one or more functions named `dmNN` or `dm_NN` (e.g. `dm01`, `dm_01`) holding a lesson's examples; `__main__` calls them in sequence.
- Content is tutorial prose + print statements, not production code. Comments explain the API (`torch.tensor`, `torch.ones/zeros/full`, `*_like` variants, NumPy↔Tensor conversion).
- `课件.zip` (~50 MB) is courseware and unrelated to the scripts; ignore it.