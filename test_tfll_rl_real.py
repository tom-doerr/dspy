#!/usr/bin/env python3
"""Test TFLL RL with REAL API calls."""

import os
import sys
import dspy
from dspy.metrics.tfll import TFLLMetric
from dspy.teleprompt.tfll_rl import TFLLRLOptimizer
import logging
import time

logging.basicConfig(level=logging.INFO)

if not os.getenv("TOGETHER_API_KEY"):
    print("ERROR: Set TOGETHER_API_KEY env variable")
    sys.exit(1)

# Setup Together API (use together_ai/ prefix for litellm)
lm = dspy.LM(
    model="together_ai/Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=os.getenv("TOGETHER_API_KEY"),
    max_tokens=50
)
dspy.settings.configure(lm=lm)

# Create REAL TFLL metric (use same model string)
tfll_metric = TFLLMetric(
    raw_chat=lm.raw_chat,
    model="together_ai/Qwen/Qwen2.5-Coder-32B-Instruct"
)

# Simple task
class QATask(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = dspy.Signature("question -> answer")

# Training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="Capital of France?", answer="Paris"),
]

print("REAL TFLL RL Test - This WILL make API calls!")
print("-"*50)

# Create optimizer
optimizer = TFLLRLOptimizer(
    metric=tfll_metric,
    num_updates=2
)

# Run optimization
program = QATask()
print(f"Initial: {program.signature.instructions}")

start = time.time()
optimized = optimizer.compile(program, trainset=trainset)
elapsed = time.time() - start

print(f"Final: {optimized.signature.instructions}")
print(f"Time: {elapsed:.2f}s (REAL API calls!)")