"""Tests for TFLL-based Reinforcement Learning optimizer."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

import dspy
from dspy.teleprompt.tfll_rl import TFLLRLOptimizer, Trajectory
from dspy.teleprompt.tfll_rl_env import DSPyEnvironment, BatchEnvironment
from dspy.metrics.tfll import TFLLMetric
from dspy.primitives import Example


class TestTrajectory:
    """Test the Trajectory class."""
    
    def test_trajectory_add_and_compute_returns(self):
        """Test adding steps and computing discounted returns."""
        traj = Trajectory()
        
        # Add some steps
        traj.add(state={"test": 1}, action="action1", reward=1.0, logprob=-0.5, value=0.8)
        traj.add(state={"test": 2}, action="action2", reward=2.0, logprob=-0.3, value=1.5)
        traj.add(state={"test": 3}, action="action3", reward=3.0, logprob=-0.2, value=2.0)
        
        # Compute returns with gamma=0.9
        returns = traj.compute_returns(gamma=0.9)
        
        # Expected returns (computed backwards):
        # R[2] = 3.0
        # R[1] = 2.0 + 0.9 * 3.0 = 4.7
        # R[0] = 1.0 + 0.9 * 4.7 = 5.23
        expected = np.array([5.23, 4.7, 3.0])
        np.testing.assert_allclose(returns, expected, rtol=1e-5)
        
    def test_trajectory_compute_advantages(self):
        """Test computing GAE advantages."""
        traj = Trajectory()
        
        # Add steps with known values
        traj.add(state={}, action="a1", reward=1.0, logprob=-0.5, value=0.5)
        traj.add(state={}, action="a2", reward=2.0, logprob=-0.3, value=1.0)
        
        # Compute advantages
        advantages = traj.compute_advantages(gamma=0.9, lam=0.95)
        
        # Advantages should be returns - values
        returns = traj.compute_returns(gamma=0.9)
        expected = returns - np.array([0.5, 1.0])
        np.testing.assert_allclose(advantages, expected, rtol=1e-5)


class TestTFLLRLOptimizer:
    """Test the TFLL RL Optimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        mock_metric = Mock(spec=TFLLMetric)
        
        optimizer = TFLLRLOptimizer(
            metric=mock_metric,
            gamma=0.95,
            episodes_per_update=5,
            num_updates=10
        )
        
        assert optimizer.gamma == 0.95
        assert optimizer.episodes_per_update == 5
        assert optimizer.num_updates == 10
        assert optimizer.metric == mock_metric
        
    def test_modify_prompt(self):
        """Test prompt modification actions."""
        mock_metric = Mock(spec=TFLLMetric)
        optimizer = TFLLRLOptimizer(metric=mock_metric)
        
        # Create a mock program with signature
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Original instructions."
        
        # Test adding chain of thought
        modified = optimizer._modify_prompt(program, "add_chain_of_thought")
        assert "step-by-step" in modified.signature.instructions.lower()
        
        # Test adding conciseness
        modified = optimizer._modify_prompt(program, "add_be_concise")
        assert "concise" in modified.signature.instructions.lower()
        
    def test_get_state(self):
        """Test state extraction from program."""
        mock_metric = Mock(spec=TFLLMetric)
        optimizer = TFLLRLOptimizer(metric=mock_metric)
        
        # Create mock program
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Think step by step. Use chain of thought reasoning."
        
        state = optimizer._get_state(program, None)
        
        assert state["has_step_by_step"] == True
        assert state["has_cot"] == True
        assert state["num_instructions"] == 2
        
    def test_select_action(self):
        """Test action selection."""
        mock_metric = Mock(spec=TFLLMetric)
        optimizer = TFLLRLOptimizer(metric=mock_metric, seed=42)
        
        # State without COT should prefer adding it
        state = {
            "instructions": "Simple instruction",
            "num_instructions": 1,
            "has_cot": False,
            "has_step_by_step": False
        }
        
        # Run multiple times to check it returns valid actions
        for _ in range(10):
            action = optimizer._select_action(state)
            assert action in optimizer.action_space
            
    def test_baseline_update(self):
        """Test baseline value updates."""
        mock_metric = Mock(spec=TFLLMetric)
        optimizer = TFLLRLOptimizer(
            metric=mock_metric,
            baseline_type="mean"
        )
        
        # Update with some returns
        returns = np.array([1.0, 2.0, 3.0])
        optimizer._update_baseline(returns)
        
        assert optimizer.baseline_value == 2.0  # Mean of [1, 2, 3]
        
    @patch('dspy.teleprompt.tfll_rl.metric_only_mode')
    def test_collect_trajectory(self, mock_metric_mode):
        """Test trajectory collection."""
        # Setup mock metric
        mock_metric = Mock(spec=TFLLMetric)
        mock_metric.return_value = -0.5  # Log probability
        
        optimizer = TFLLRLOptimizer(
            metric=mock_metric,
            max_prompt_modifications=3
        )
        
        # Create mock program
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Initial instructions"
        
        # Create examples
        examples = [
            Example(input="test1", label="answer1"),
            Example(input="test2", label="answer2")
        ]
        
        # Collect trajectory
        trajectory = optimizer._collect_trajectory(program, examples, max_steps=2)
        
        assert len(trajectory.states) == 2
        assert len(trajectory.actions) == 2
        assert len(trajectory.rewards) == 2
        assert len(trajectory.logprobs) == 2
        
    @patch('dspy.teleprompt.tfll_rl.metric_only_mode')
    @patch('dspy.teleprompt.tfll_rl.logger')
    def test_compile(self, mock_logger, mock_metric_mode):
        """Test the compile method."""
        # Setup mock metric
        mock_metric = Mock(spec=TFLLMetric)
        mock_metric.return_value = -0.3
        
        optimizer = TFLLRLOptimizer(
            metric=mock_metric,
            num_updates=2,
            episodes_per_update=2
        )
        
        # Create mock student program
        student = Mock()
        student.signature = Mock()
        student.signature.instructions = "Original"
        
        # Create trainset
        trainset = [
            Example(input="q1", label="a1"),
            Example(input="q2", label="a2")
        ]
        
        # Run compile
        optimized = optimizer.compile(student, trainset=trainset)
        
        # Should return a modified program
        assert optimized is not None
        assert hasattr(optimized, 'signature')
        
        # Should log progress
        assert mock_logger.info.called


class TestDSPyEnvironment:
    """Test the DSPy Gym environment."""
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Initial"
        
        trainset = [Example(input="test", label="answer")]
        metric = Mock(spec=TFLLMetric)
        action_space = ["action1", "action2"]
        modify_fn = Mock()
        
        env = DSPyEnvironment(
            program=program,
            trainset=trainset,
            metric=metric,
            action_space=action_space,
            modify_prompt_fn=modify_fn
        )
        
        assert env.action_space.n == 2
        assert env.max_steps == 10
        
    @patch('dspy.teleprompt.tfll_rl_env.metric_only_mode')
    def test_environment_reset(self, mock_metric_mode):
        """Test environment reset."""
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Test"
        
        trainset = [Example(input="q", label="a")]
        metric = Mock(spec=TFLLMetric, return_value=0.5)
        
        env = DSPyEnvironment(
            program=program,
            trainset=trainset,
            metric=metric,
            action_space=["a1"],
            modify_prompt_fn=Mock()
        )
        
        obs = env.reset()
        
        assert env.current_step == 0
        assert env.current_example is not None
        assert "instructions_length" in obs
        assert "has_cot" in obs
        
    @patch('dspy.teleprompt.tfll_rl_env.metric_only_mode')
    def test_environment_step(self, mock_metric_mode):
        """Test taking a step in the environment."""
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Initial"
        
        trainset = [Example(input="q", label="a")]
        metric = Mock(spec=TFLLMetric, return_value=0.7)
        
        modified_program = Mock()
        modified_program.signature = Mock()
        modified_program.signature.instructions = "Modified"
        modify_fn = Mock(return_value=modified_program)
        
        env = DSPyEnvironment(
            program=program,
            trainset=trainset,
            metric=metric,
            action_space=["modify"],
            modify_prompt_fn=modify_fn,
            max_steps=5
        )
        
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert env.current_step == 1
        assert modify_fn.called
        assert "current_score" in info
        assert not truncated  # Not at max steps yet
        
    def test_environment_render(self):
        """Test environment rendering."""
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Test instructions"
        
        env = DSPyEnvironment(
            program=program,
            trainset=[Example(input="q", label="a")],
            metric=Mock(spec=TFLLMetric),
            action_space=["a"],
            modify_prompt_fn=Mock()
        )
        
        # Test string rendering
        output = env.render(mode="string")
        assert "Step:" in output
        assert "Instructions:" in output
        
    def test_batch_environment(self):
        """Test batch environment for parallel execution."""
        program = Mock()
        program.signature = Mock()
        program.signature.instructions = "Test"
        
        trainset = [Example(input=f"q{i}", label=f"a{i}") for i in range(5)]
        metric = Mock(spec=TFLLMetric, return_value=0.5)
        
        batch_env = BatchEnvironment(
            num_envs=3,
            program=program,
            trainset=trainset,
            metric=metric,
            action_space=["a1", "a2"],
            modify_prompt_fn=Mock()
        )
        
        assert len(batch_env.envs) == 3
        
        # Test batch reset
        observations = batch_env.reset()
        assert len(observations) == 3
        
        # Test batch step
        actions = [0, 1, 0]
        with patch('dspy.teleprompt.tfll_rl_env.metric_only_mode'):
            obs, rewards, terms, truncs, infos = batch_env.step(actions)
        
        assert len(obs) == 3
        assert len(rewards) == 3
        assert len(infos) == 3


class TestIntegration:
    """Integration tests for TFLL RL optimizer with environment."""
    
    @patch('dspy.teleprompt.tfll_rl.metric_only_mode')
    def test_optimizer_with_gym_interface(self, mock_metric_mode):
        """Test optimizer using the Gym interface."""
        # Create a simple DSPy program
        class SimpleProgram(dspy.Module):
            def __init__(self):
                super().__init__()
                self.signature = dspy.Signature("input -> output")
                self.signature.instructions = "Answer the question"
                
        program = SimpleProgram()
        
        # Setup metric
        mock_metric = Mock(spec=TFLLMetric)
        mock_metric.return_value = -0.5
        
        # Create optimizer with Gym interface
        optimizer = TFLLRLOptimizer(
            metric=mock_metric,
            use_gym_interface=True,
            num_updates=1,
            episodes_per_update=1
        )
        
        trainset = [Example(input="What is 2+2?", output="4")]
        
        # Create environment
        env = optimizer.make_env(program, trainset)
        
        assert env is not None
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        
        # Run a simple episode
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 3:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
        assert steps > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])