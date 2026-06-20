# Speculative Decoding Implementation

A PyTorch implementation of [speculative decoding](https://arxiv.org/abs/2211.17192) — a technique for accelerating autoregressive language model inference by using a small draft model to propose tokens and a large target model to verify them in parallel.

## Overview

Standard autoregressive decoding generates one token at a time, making it bottlenecked by sequential forward passes. Speculative decoding breaks this bottleneck:

1. **Draft phase**: A small, fast model proposes K tokens via top-k sampling
2. **Verify phase**: A large target model evaluates all K tokens in a single forward pass
3. **Accept/Reject**: Each token is accepted with probability `min(1, p(x)/q(x))`, where:
   - `p(x)` = target model probability for token x
   - `q(x)` = draft model probability for token x
4. On rejection, resample from adjusted distribution `max(0, p(x) - q(x))`

This produces **exact samples from the target distribution** while being significantly faster.

## Quick Start

```bash
# Install dependencies
uv sync

# Run speculative decoding
cd speculative-decoding
uv run python -c "
import sys
sys.path.insert(0, 'src')

from src.models import LanguageModel
from src.decoding import SpeculativeDecoder

draft = LanguageModel('HuggingFaceTB/SmolLM-135M')
target = LanguageModel('HuggingFaceTB/SmolLM-135M')

decoder = SpeculativeDecoder(draft, target, k=3, verbose=True)
token_ids, text, stats = decoder.generate('The capital of France is', max_new_tokens=50)

print(f'OUTPUT: {text}')
print(f'STATS:  {stats}')
print(f'Acceptance rate: {stats.accepted_tokens / stats.drafted_tokens:.1%}')
"
```

## Project Structure

```
speculative-decoding/
├── src/
│   ├── decoding/
│   │   ├── __init__.py
│   │   ├── greedy.py          # Greedy decoding baseline
│   │   ├── sampling.py        # Top-k, top-p, temperature sampling
│   │   └── speculative.py     # Speculative decoding implementation
│   └── models/
│       ├── loader.py          # HuggingFace model loading utilities
│       ├── tokenizer.py       # Tokenizer wrapper
│       └── wrapper.py         # LanguageModel/EmbeddingModel abstractions
├── tests/
│   ├── test_greedy.py
│   ├── test_sampling.py
│   └── test_speculative.py    # E2E tests for speculative decoding
├── configs/
├── data/
├── report/
└── results/
```

## Core Components

### `SpeculativeDecoder`

```python
class SpeculativeDecoder:
    def __init__(
        self,
        draft_model: LanguageModel,    # Small, fast model
        target_model: LanguageModel,   # Large, accurate model
        k: int = 3,                    # Tokens to draft per round
        temperature: float = 1.0,      # Sampling temperature
        top_k: int = 50,               # Top-k filtering
        verbose: bool = False,         # Enable step-by-step logs
    )
```

**Methods:**
- `draft_step()` — Generate K candidate tokens with draft model
- `verify_step()` — Get target probabilities for draft tokens
- `acceptance_step()` — Accept/reject using min(1, p/q) criterion
- `generate()` — Main loop until EOS

### `DecodeStats`

```python
@dataclass
class DecodeStats:
    drafted_tokens: int   # Total tokens proposed by draft model
    accepted_tokens: int  # Tokens accepted (matching or probabilistically)
    rejected_tokens: int  # Tokens rejected
    target_calls: int     # Number of target model forward passes
```

## Algorithm Details

### Acceptance Criterion

For each draft token x with draft probability q(x) and target probability p(x):

```
accept_ratio = min(1, p(x) / q(x))
u ~ Uniform(0, 1)

if u <= accept_ratio:
    accept token x
else:
    reject and resample from adjusted distribution
```

### Resampling on Rejection

When a token is rejected, we sample from:

```
p'(x) = normalize(max(0, p(x) - q(x)))
```

This ensures the overall distribution matches the target model exactly.

### Bonus Token

When all K draft tokens are accepted, we sample one additional token from the target model to maintain the correct output length.

## Verbose Output

With `verbose=True`, you can see the algorithm in action:

```
PROMPT:   'The capital of France is'
----------------------------------------
DRAFT:    ' a long time'
TARGET:   ' Paris city,'
ACCEPTED: ' a' (p=0.0132, q=0.0150, ratio=0.8821, u=0.5939)
ACCEPTED: ' long' (p=0.0031, q=0.0062, ratio=0.4926, u=0.0829)
ACCEPTED: ' time' (p=0.0245, q=0.0315, ratio=0.7791, u=0.6482)
BONUS:    ' and' (from target)
----------------------------------------
DRAFT:    ' was founded in'
TARGET:   ' a the in'
ACCEPTED: ' was' (p=0.0066, q=0.0135, ratio=0.4887, u=0.3508)
REJECTED: ' founded' (p=0.0486, q=0.0757, ratio=0.6419, u=0.8983)
  (resampled target token: ' once')
----------------------------------------
```

## Running Tests

```bash
# All tests
uv run python speculative-decoding/tests/test_speculative.py

# Sampling tests
uv run python speculative-decoding/tests/test_sampling.py

# Greedy tests
uv run python speculative-decoding/tests/test_greedy.py
```

## Dependencies

- Python >= 3.12
- PyTorch >= 2.0
- Transformers >= 4.40
- Accelerate >= 0.27

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2022
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al., 2023
