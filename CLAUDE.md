# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is DSPy

DSPy (Declarative Self-improving Python) is a framework for programming—not prompting—language models. It allows developers to build modular AI systems with composable Python code and automatically optimize prompts and weights.

## Key Development Commands

### Running Tests
```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/predict

# With uv (recommended)
uv run pytest tests/predict

# Run a single test
pytest tests/predict/test_chain_of_thought.py::test_cot_initialization
```

### Code Quality
```bash
# Run pre-commit hooks (includes ruff)
pre-commit run --all-files

# Run ruff directly
ruff check dspy/
ruff format dspy/

# Install pre-commit hooks for automatic checking
pre-commit install
```

### Building and Installation
```bash
# Install in development mode with uv (recommended)
uv sync --extra dev

# Alternative: standard pip install
pip install -e ".[dev]"

# Build package
python -m build
```

## Architecture Overview

### Core Components Hierarchy
1. **Signatures** (`dspy/signatures/`) - Define structured I/O schemas (e.g., "question -> answer")
2. **Modules** (`dspy/primitives/module.py`) - Base class for all DSPy components
3. **Predictors** (`dspy/predict/`) - Implement various prompting strategies (Predict, ChainOfThought, ReAct)
4. **Adapters** (`dspy/adapters/`) - Convert between signatures and LLM-specific formats
5. **Clients** (`dspy/clients/`) - Handle LLM communication (OpenAI, Anthropic, local models)
6. **Optimizers** (`dspy/teleprompt/`) - Improve prompts/weights (BootstrapFewShot, MIPRO, SIMBA)

### Key Design Patterns

**Module Composition**: All DSPy components inherit from `dspy.Module` and implement:
- `__call__()` for synchronous execution
- `acall()` for asynchronous execution
- `forward()` method containing the main logic

**Signature Processing**: The flow is:
1. User defines signature (e.g., `"context, question -> answer"`)
2. Signature parser creates structured fields with types/descriptions
3. Adapter formats prompt based on signature and LLM requirements
4. Client sends to LLM and receives response
5. Adapter parses response back to structured output

**Optimization Loop**: Optimizers work by:
1. Collecting examples through bootstrapping or provided data
2. Generating candidate prompts/demonstrations
3. Evaluating performance with metrics
4. Selecting best performing configurations

### Important Implementation Details

**Async Support**: Most modules support both sync and async execution. Always check if a module has `acall()` method before using it asynchronously.

**Caching**: DSPy uses `diskcache` for caching LLM responses. Cache is stored in `~/.dspy/`.

**Settings Management**: Global LM configuration via `dspy.settings.configure(lm=...)`. Context managers available for temporary settings.

**Type Handling**: DSPy supports multimodal inputs (images, audio) through the adapter system. Check `dspy/adapters/types/` for supported types.

## Common Development Patterns

### Creating a New Predictor Module
New predictors should:
1. Inherit from `dspy.Module`
2. Accept a signature in `__init__`
3. Implement `forward()` method
4. Use `dspy.Predict` or other primitives internally
5. Support both sync and async if possible

### Adding a New Optimizer
Optimizers should:
1. Inherit from `teleprompt.Teleprompter`
2. Implement `compile()` method
3. Return an optimized copy of the input module
4. Use `dspy.evaluate` utilities for scoring

### Working with Adapters
When modifying adapters:
1. Check `dspy/adapters/base_adapter.py` for the interface
2. Ensure compatibility with all output formats (Chat, JSON, XML)
3. Handle both single and batch predictions
4. Preserve streaming capabilities where applicable

## Testing Conventions

- Unit tests mirror source structure in `tests/`
- Use pytest fixtures for common setups
- Mock LLM calls when testing logic
- Integration tests should use real models sparingly
- Reliability tests in `tests/reliability/` for complex scenarios

## Important Notes

- Always preserve backward compatibility when modifying core modules
- Signature syntax is fundamental - changes affect entire ecosystem
- Optimizers should be model-agnostic when possible
- Cache invalidation is handled automatically for most changes
- Documentation in `docs/` uses MkDocs - preview with `mkdocs serve`

## Custom Fork Features

### SIMBAT Optimizer
- Location: `dspy/teleprompt/simbat.py`
- SIMBA with Tail evaluation for better optimization
- Usage: `--optimizer simbat --tail-eval-size 128`
- Requires Python 3.11+

### TFLL Metric (Completed)
- Location: `dspy/metrics/tfll.py`
- Teacher-Forced Log-Likelihood metric
- One raw API call with echo+logprobs
- For metric-only optimization
- Supports margin calculation (TFLL-M)
- Generates feedback for GEPA optimization

## TFLL Implementation Details

### Files Added/Modified
- `dspy/metrics/tfll.py` - Complete TFLL metric implementation
- `dspy/metrics/__init__.py` - Exports TFLLMetric
- `dspy/clients/lm.py` - Added `raw_chat()` method
- `dspy/utils/metric_only.py` - NullLM and metric_only_mode context

### Usage Example
```python
from dspy.metrics.tfll import TFLLMetric
from dspy.utils.metric_only import metric_only_mode
import dspy

# Setup TFLL metric
lm = dspy.settings.lm
metric = TFLLMetric(
    raw_chat=lm.raw_chat,
    model="openrouter/google/gemini-2.5-flash",
    use_margin=True,
    margin_alpha=0.5,
    top_logprobs=5
)

# Use with optimizer (no generation costs)
with metric_only_mode():
    best = optimizer.optimize(program, trainset, metric)
```

### Key Features
- **Zero generation cost**: Uses teacher-forced labels with echo mode
- **Single API call**: One request per example for scoring
- **Margin support**: Optional TFLL-M with top-k alternatives
- **GEPA compatible**: Provides textual feedback for reflection
- **Drop-in integration**: Works with all existing optimizers

## TFLL-Based Reinforcement Learning (Implemented)

### Overview
TFLL-based RL optimizer implemented for policy gradient optimization, using Teacher-Forced Log-Likelihood for efficient prompt evaluation without generation costs.

### Implementation Details

#### Files Added
- `dspy/teleprompt/tfll_rl.py` - TFLLRLOptimizer with policy gradient optimization
- `dspy/teleprompt/tfll_rl_env.py` - OpenAI Gym-compatible environment wrapper
- `tests/teleprompt/test_tfll_rl.py` - Comprehensive test suite

#### Key Components

**TFLLRLOptimizer**
- Policy gradient optimization using TFLL scoring
- Trajectory collection with discounted rewards (γ parameter)
- GAE advantage estimation with baseline
- Action space: prompt modifications (add COT, step-by-step, etc.)
- **Acceptance Policy**: Accepts modifications when:
  - Policy gradient signal (advantage * logprob) is positive
  - No additional evaluation needed - uses trajectory data
- Supports both standard DSPy and Gym interfaces

**DSPyEnvironment**
- OpenAI Gym-compatible interface
- State: prompt features (length, keywords, current score)
- Actions: prompt modifications
- Rewards: TFLL score improvements + optional shaping
- Supports batch environments for parallel collection

**Usage Example**
```python
from dspy.teleprompt import TFLLRLOptimizer
from dspy.metrics.tfll import TFLLMetric

# Setup
metric = TFLLMetric(
    raw_chat=lm.raw_chat,
    model="together/Qwen/Qwen2.5-Coder-32B-Instruct"
)

# Standard DSPy interface
optimizer = TFLLRLOptimizer(
    metric=metric,
    gamma=0.99,
    num_updates=100,
    episodes_per_update=10
)
optimized = optimizer.compile(program, trainset=trainset)

# Gym interface
optimizer = TFLLRLOptimizer(metric=metric, use_gym_interface=True)
env = optimizer.make_env(program, trainset)
for episode in range(100):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
```

### Model Support for Logprobs

**Recommended Model**: For getting prompt logprobs efficiently, use:
- **Qwen/Qwen2.5-Coder-32B-Instruct** via Together AI API
- Model string: `"together/Qwen/Qwen2.5-Coder-32B-Instruct"`
- Supports echo mode with logprobs for TFLL scoring
- Good balance of cost and performance

### Features
- **Zero generation cost**: Uses TFLL echo mode for scoring
- **Gym compatibility**: Works with standard RL libraries
- **Batch processing**: Parallel environment support
- **Flexible baselines**: Mean, moving average, or custom
- **Action space**: 9 predefined prompt modifications
- **Reward shaping**: Optional complexity penalties and bonuses
- **Policy Gradient Acceptance**: Actions weighted by advantage * logprob
  - Positive PG signal → accept modification
  - Negative PG signal → reject modification
  - No need to evaluate each modification separately
  - Efficient: uses existing trajectory data

### Advantages
- Efficient policy evaluation without generation
- Handles sequential decision making naturally
- Compatible with existing RL algorithms (PPO, A2C)
- Online learning capability
- Variance reduction through baselines

### Future Enhancements
- PPO-style clipping for stability
- Custom value networks for better baselines
- Learnable action embeddings
- Integration with other DSPy optimizers
- Curriculum learning support

### Testing & Experience Replay
- Created CharCountEnv test environment (no LLM calls needed)
- Added replay buffer (10K experiences) for sample efficiency
- Stores trajectory data for reuse across updates
- SimpleMetric test env: fixed 50-char target, timestamp inputs
- Optimized for single experience collection per update  
- Uses N most recent experiences for policy evaluation
- **FIXED**: Single-step updates now working - accepts improvements immediately
- Modifications accepted when reward > 0 (9 accepts in 20 steps test run)