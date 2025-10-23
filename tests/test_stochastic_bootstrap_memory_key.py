import dspy


class EchoModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pred = dspy.Predict("a: str -> b: str")

    def forward(self, a: str):
        return self.pred(a=a)


def dummy_reward(args, pred) -> float:
    return 1.0


def test_memory_key_ignores_transport_fields():
    mod = EchoModule()
    boot = dspy.StochasticBootstrapBestOfN(mod, reward_fn=dummy_reward, N=1)

    k1 = boot._memory_key({"a": "x"})
    k2 = boot._memory_key({"a": "x", "image_path": "/tmp/p.png"})
    k3 = boot._memory_key({"a": "x", "repo_url": "https://github.com/owner/repo"})
    k4 = boot._memory_key({"a": "x", "image_path": "/tmp/p.png", "repo_url": "u"})

    assert k1 == k2 == k3 == k4


def test_memory_key_uses_signature_inputs_only():
    mod = EchoModule()
    boot = dspy.StochasticBootstrapBestOfN(mod, reward_fn=dummy_reward, N=1)

    k1 = boot._memory_key({"a": "x"})
    k2 = boot._memory_key({"a": "x", "unrelated": 123})
    assert k1 == k2


def test_memory_key_global_when_per_key_disabled():
    mod = EchoModule()
    boot = dspy.StochasticBootstrapBestOfN(mod, reward_fn=dummy_reward, N=1, per_key_memory=False)
    assert boot._memory_key({"a": "x", "image_path": "/x"}) == "global"


def test_rollout_counter_increments():
    mod = StoreModule()
    boot = dspy.StochasticBootstrapBestOfN(mod, reward_fn=dummy_reward, N=2)
    start = boot._rollout
    boot(a="x")
    mid = boot._rollout
    boot(a="y")
    end = boot._rollout
    assert mid > start
    assert end > mid


class StoreModule(dspy.Module):
    def forward(self, a: str, **kwargs):
        # return mapping compatible with _build_demo
        return {"b": a}


def test_replay_store_and_eviction():
    mod = StoreModule()
    boot = dspy.StochasticBootstrapBestOfN(mod, reward_fn=lambda args, pred: 0.5, N=1, replay_buffer_size=2)
    # Three calls should cap memory at 2
    boot(a="v1")
    boot(a="v2")
    boot(a="v3")
    snap = boot.memory_snapshot()
    # Only one key present
    assert len(snap) == 1
    (examples,) = snap.values()
    assert len(examples) == 2
