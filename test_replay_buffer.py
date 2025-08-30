from dspy.teleprompt.replay_buffer import ReplayBuffer

buf = ReplayBuffer(100)

# Add 50 experiences
for i in range(50):
    buf.add({'i': i}, f'a{i}', i*0.1, -i*0.5, i*0.2)

print(f"Size: {len(buf)}")

# Sample batch
batch = buf.sample_batch(10)
print(f"Sampled: {len(batch)}")

# Test overflow
for i in range(100):
    buf.add({}, 'x', 0, 0, 0)

print(f"Final: {len(buf)} (max 100)")