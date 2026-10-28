#!/usr/bin/env python3
"""Run the complete correctness suite on CPU from the repository root.

Pass pytest options to select checks or change report paths. With no options,
run every test and write validation/fixed-suite.xml.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ["ACCELERATE_USE_CPU"] = "true"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    import pytest

    # The loader otherwise selects MPS on macOS even with ACCELERATE_USE_CPU.
    torch.backends.mps.is_available = lambda: False
    args = sys.argv[1:] or [
        "-q", "-s", "speculative-decoding/tests",
        "--junitxml=validation/fixed-suite.xml",
    ]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
