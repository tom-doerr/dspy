"""OpenAI Gym-compatible environment for DSPy programs."""

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import dspy
from dspy.primitives import Example, Module
from dspy.metrics.tfll import TFLLMetric
from dspy.utils.metric_only import metric_only_mode


class DSPyEnvironment:
    """
    OpenAI Gym-compatible environment for DSPy program optimization.
    
    This environment treats DSPy programs as agents that can be modified
    through actions (prompt modifications) and evaluated using metrics.
    """
    
    def __init__(
        self,
        program: Module,
        trainset: List[Example],
        metric: TFLLMetric,
        action_space: List[str],
        modify_prompt_fn: Callable[[Module, str], Module],
        max_steps: int = 10,
        reward_shaping: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the DSPy environment.
        
        Args:
            program: Base DSPy program to optimize
            trainset: Training examples
            metric: Metric for evaluation (typically TFLL)
            action_space: List of possible actions
            modify_prompt_fn: Function to apply actions to program
            max_steps: Maximum steps per episode
            reward_shaping: Whether to use reward shaping
            seed: Random seed
        """
        self.base_program = copy.deepcopy(program)
        self.current_program = None
        self.trainset = trainset
        self.metric = metric
        self.action_space_list = action_space
        self.modify_prompt_fn = modify_prompt_fn
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        
        # Initialize random generators
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Episode tracking
        self.current_step = 0
        self.current_example = None
        self.episode_rewards = []
        self.previous_score = None
        
        # Action and observation spaces (Gym-compatible)
        self.action_space = DiscreteSpace(len(action_space))
        self.observation_space = DictSpace({
            "instructions_length": BoxSpace(0, 1000, shape=(1,)),
            "num_sentences": BoxSpace(0, 50, shape=(1,)),
            "has_cot": BoxSpace(0, 1, shape=(1,)),
            "has_step_by_step": BoxSpace(0, 1, shape=(1,)),
            "current_score": BoxSpace(-10, 10, shape=(1,)),
        })
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Optional seed for reproducibility
            
        Returns:
            Initial observation
        """
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)
            
        # Reset program to base state
        self.current_program = copy.deepcopy(self.base_program)
        
        # Sample new example
        self.current_example = self.rng.choice(self.trainset)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.previous_score = self._evaluate_current_program()
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if action < 0 or action >= len(self.action_space_list):
            raise ValueError(f"Invalid action: {action}")
            
        # Apply action
        action_name = self.action_space_list[action]
        self.current_program = self.modify_prompt_fn(self.current_program, action_name)
        
        # Evaluate new program
        current_score = self._evaluate_current_program()
        
        # Compute reward
        reward = self._compute_reward(current_score)
        self.episode_rewards.append(reward)
        
        # Update tracking
        self.current_step += 1
        self.previous_score = current_score
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "action_taken": action_name,
            "current_score": current_score,
            "episode_step": self.current_step,
            "cumulative_reward": sum(self.episode_rewards),
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ("human" or "string")
            
        Returns:
            String representation if mode is "string"
        """
        instructions = ""
        if hasattr(self.current_program, "signature"):
            if hasattr(self.current_program.signature, "instructions"):
                instructions = self.current_program.signature.instructions or ""
                
        output = f"Step: {self.current_step}/{self.max_steps}\n"
        output += f"Current Instructions: {instructions[:100]}...\n"
        if self.previous_score is not None:
            output += f"Previous Score: {self.previous_score:.4f}\n"
        else:
            output += "Previous Score: None\n"
        output += f"Episode Rewards: {self.episode_rewards}\n"
        
        if mode == "human":
            print(output)
            return None
        else:
            return output
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from the environment state."""
        instructions = ""
        if hasattr(self.current_program, "signature"):
            if hasattr(self.current_program.signature, "instructions"):
                instructions = self.current_program.signature.instructions or ""
        
        # Handle Mock objects in tests
        if isinstance(instructions, str):
            inst_len = len(instructions)
            num_sentences = len(instructions.split("."))
            has_cot = 1.0 if "chain of thought" in instructions.lower() else 0.0
            has_step = 1.0 if "step by step" in instructions.lower() else 0.0
        else:
            inst_len = 0
            num_sentences = 0
            has_cot = 0.0
            has_step = 0.0
                
        return {
            "instructions_length": np.array([inst_len], dtype=np.float32),
            "num_sentences": np.array([num_sentences], dtype=np.float32),
            "has_cot": np.array([has_cot], dtype=np.float32),
            "has_step_by_step": np.array([has_step], dtype=np.float32),
            "current_score": np.array([self.previous_score if self.previous_score is not None else 0.0], dtype=np.float32),
        }
    
    def _evaluate_current_program(self) -> float:
        """Evaluate the current program using the metric."""
        with metric_only_mode():
            # Evaluate on current example
            score = self.metric(self.current_example, self.current_program)
            
            # Optionally evaluate on a small batch for more stable rewards
            if len(self.trainset) > 1:
                batch_size = min(3, len(self.trainset))
                batch = self.rng.sample(self.trainset, batch_size)
                batch_score = sum(self.metric(ex, self.current_program) for ex in batch)
                score = (score + batch_score) / (batch_size + 1)
                
        return float(score)
    
    def _compute_reward(self, current_score: float) -> float:
        """
        Compute reward based on score change and optional shaping.
        
        Args:
            current_score: Current program score
            
        Returns:
            Computed reward
        """
        if self.previous_score is None:
            base_reward = current_score
        else:
            # Reward is improvement in score
            base_reward = current_score - self.previous_score
            
        if self.reward_shaping:
            # Add shaping based on program complexity
            instructions = ""
            if hasattr(self.current_program, "signature"):
                if hasattr(self.current_program.signature, "instructions"):
                    instructions = self.current_program.signature.instructions or ""
                    
            # Penalty for too long instructions
            # Handle case where instructions might be a Mock object in tests
            if isinstance(instructions, str):
                length_penalty = -0.001 * max(0, len(instructions) - 200)
                
                # Bonus for including key phrases
                bonus = 0
                if "step by step" in instructions.lower():
                    bonus += 0.1
                if "chain of thought" in instructions.lower():
                    bonus += 0.1
            else:
                length_penalty = 0
                bonus = 0
                
            return base_reward + length_penalty + bonus
        else:
            return base_reward
    
    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate early.
        
        Returns:
            True if episode should terminate
        """
        # Could add early stopping conditions
        # For example, if score is very high or very low
        if self.previous_score is not None:
            if self.previous_score > 0.9:  # Very good score
                return True
            if self.previous_score < -5.0:  # Very bad score
                return True
        return False


class DiscreteSpace:
    """Simple discrete action space."""
    
    def __init__(self, n: int):
        self.n = n
        
    def sample(self) -> int:
        return random.randint(0, self.n - 1)
    
    def contains(self, x: int) -> bool:
        return 0 <= x < self.n


class BoxSpace:
    """Simple continuous observation space."""
    
    def __init__(self, low: float, high: float, shape: Tuple[int, ...]):
        self.low = low
        self.high = high
        self.shape = shape
        
    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape)
    
    def contains(self, x: np.ndarray) -> bool:
        return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)


class DictSpace:
    """Dictionary observation space."""
    
    def __init__(self, spaces: Dict[str, BoxSpace]):
        self.spaces = spaces
        
    def sample(self) -> Dict[str, np.ndarray]:
        return {key: space.sample() for key, space in self.spaces.items()}
    
    def contains(self, x: Dict[str, np.ndarray]) -> bool:
        return all(key in x and space.contains(x[key]) for key, space in self.spaces.items())


class BatchEnvironment:
    """
    Vectorized environment for parallel trajectory collection.
    
    This allows running multiple environment instances in parallel
    for more efficient data collection.
    """
    
    def __init__(
        self,
        num_envs: int,
        program: Module,
        trainset: List[Example],
        metric: TFLLMetric,
        action_space: List[str],
        modify_prompt_fn: Callable[[Module, str], Module],
        **env_kwargs
    ):
        """
        Initialize batch environment.
        
        Args:
            num_envs: Number of parallel environments
            program: Base DSPy program
            trainset: Training examples
            metric: Evaluation metric
            action_space: List of possible actions
            modify_prompt_fn: Function to apply actions
            **env_kwargs: Additional arguments for DSPyEnvironment
        """
        self.num_envs = num_envs
        self.envs = [
            DSPyEnvironment(
                program=program,
                trainset=trainset,
                metric=metric,
                action_space=action_space,
                modify_prompt_fn=modify_prompt_fn,
                seed=i,
                **env_kwargs
            )
            for i in range(num_envs)
        ]
        
    def reset(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Reset all environments."""
        return [env.reset(seed=seed + i if seed else None) for i, env in enumerate(self.envs)]
    
    def step(self, actions: List[int]) -> Tuple[List[Dict[str, Any]], List[float], List[bool], List[bool], List[Dict[str, Any]]]:
        """Take a step in all environments."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, terminateds, truncateds, infos = zip(*results)
        return list(observations), list(rewards), list(terminateds), list(truncateds), list(infos)
    
    def render(self, mode: str = "human"):
        """Render all environments."""
        for i, env in enumerate(self.envs):
            print(f"Environment {i}:")
            env.render(mode)
            
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()