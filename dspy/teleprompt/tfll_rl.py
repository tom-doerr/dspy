"""TFLL-based Reinforcement Learning Optimizer for DSPy."""

import copy
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

import dspy
from dspy.primitives import Example, Module
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.metrics.tfll import TFLLMetric
from dspy.utils.metric_only import metric_only_mode
from dspy.evaluate import Evaluate
from dspy.teleprompt.experience_prompt_updater import RuleBasedUpdater

logger = logging.getLogger(__name__)


class Trajectory:
    """Stores a trajectory of (state, action, reward, logprob) tuples."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []  # For baseline/value function
        
    def add(self, state, action, reward, logprob, value=0.0):
        """Add a step to the trajectory."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logprobs.append(logprob)
        self.values.append(value)
        
    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns for each timestep."""
        returns = np.zeros(len(self.rewards))
        running_return = 0
        for t in reversed(range(len(self.rewards))):
            running_return = self.rewards[t] + gamma * running_return
            returns[t] = running_return
        return returns
    
    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
        """Compute GAE advantages."""
        returns = self.compute_returns(gamma)
        values = np.array(self.values)
        advantages = returns - values
        return advantages


class ExperienceBuffer:
    """Buffer for storing and reusing trajectory experiences."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize experience buffer."""
        from collections import deque
        self.buffer = deque(maxlen=max_size)
        self.total_added = 0


class TFLLRLOptimizer(Teleprompter):
    """
    TFLL-based Reinforcement Learning optimizer for DSPy.
    
    Uses Teacher-Forced Log-Likelihood for efficient policy evaluation
    without generation costs, implementing policy gradient optimization.
    """
    
    def __init__(
        self,
        metric: Optional[TFLLMetric] = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        episodes_per_update: int = 1,
        num_updates: int = 100,
        learning_rate: float = 0.01,
        baseline_type: str = "mean",  # "mean", "moving_average", or "value_network"
        advantage_threshold: float = 0.0,
        max_prompt_modifications: int = 5,
        temperature: float = 1.0,
        seed: int = 0,
        num_threads: int = 1,
        use_gym_interface: bool = False,
        use_experience_replay: bool = True,
        buffer_size: int = 10000,
    ):
        """
        Initialize the TFLL RL optimizer.
        
        Args:
            metric: TFLL metric instance for scoring
            gamma: Discount factor for future rewards
            lam: Lambda for GAE advantage estimation
            episodes_per_update: Number of episodes to collect before policy update (default: 1)
            num_updates: Total number of policy updates
            learning_rate: Learning rate for policy updates
            baseline_type: Type of baseline for variance reduction
            advantage_threshold: Minimum advantage to accept prompt modification
            max_prompt_modifications: Maximum prompt modifications per episode
            temperature: Temperature for action sampling
            seed: Random seed
            num_threads: Number of threads for parallel evaluation
            use_gym_interface: Whether to use Gym-compatible interface
        """
        super().__init__()
        self.metric = metric
        self.gamma = gamma
        self.lam = lam
        self.episodes_per_update = episodes_per_update
        self.num_updates = num_updates
        self.learning_rate = learning_rate
        self.baseline_type = baseline_type
        self.advantage_threshold = advantage_threshold
        self.max_prompt_modifications = max_prompt_modifications
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.num_threads = num_threads
        self.use_gym_interface = use_gym_interface
        self.use_experience_replay = use_experience_replay
        
        # Experience replay buffer
        if use_experience_replay:
            from dspy.teleprompt.replay_buffer import ReplayBuffer
            self.replay_buffer = ReplayBuffer(buffer_size)
        else:
            self.replay_buffer = None
        
        # Baseline tracking
        self.baseline_value = 0.0
        self.baseline_alpha = 0.1  # For moving average
        
        # Experience-based prompt updater
        self.prompt_updater = RuleBasedUpdater()
        
        # Action space: possible prompt modifications
        self.action_space = [
            "add_chain_of_thought",
            "add_step_by_step",
            "add_lets_think",
            "add_be_concise",
            "add_be_detailed",
            "remove_instruction",
            "simplify_instruction",
            "emphasize_accuracy",
            "emphasize_reasoning",
        ]
        
    def _modify_prompt(self, program: Module, action: str) -> Module:
        """
        Apply an action to modify the program's prompt.
        
        Args:
            program: Current DSPy program
            action: Action to apply
            
        Returns:
            Modified program copy
        """
        modified = copy.deepcopy(program)
        
        # Get current instructions
        instructions = ""
        if hasattr(modified, "signature") and hasattr(modified.signature, "instructions"):
            instructions = modified.signature.instructions or ""
        
        # Apply action
        if action == "add_chain_of_thought":
            instructions += "\nLet's approach this step-by-step."
        elif action == "add_step_by_step":
            instructions += "\nThink through this systematically, step by step."
        elif action == "add_lets_think":
            instructions += "\nLet's think about this carefully."
        elif action == "add_be_concise":
            instructions += "\nBe concise and direct in your response."
        elif action == "add_be_detailed":
            instructions += "\nProvide a detailed and thorough response."
        elif action == "remove_instruction" and instructions:
            # Remove last sentence
            sentences = instructions.split(".")
            if len(sentences) > 1:
                instructions = ".".join(sentences[:-1]) + "."
        elif action == "simplify_instruction":
            # Simplify by removing adjectives (basic implementation)
            instructions = instructions.replace("carefully", "").replace("thoroughly", "")
            instructions = instructions.replace("detailed", "").replace("  ", " ")
        elif action == "emphasize_accuracy":
            instructions += "\nFocus on accuracy and correctness."
        elif action == "emphasize_reasoning":
            instructions += "\nExplain your reasoning clearly."
            
        # Update instructions
        if hasattr(modified, "signature") and hasattr(modified.signature, "instructions"):
            modified.signature.instructions = instructions.strip()
            
        return modified
    
    def _collect_trajectory(
        self, 
        program: Module, 
        examples: List[Example],
        max_steps: Optional[int] = None
    ) -> Trajectory:
        """
        Collect a trajectory by modifying prompts and evaluating with TFLL.
        
        Args:
            program: Initial program
            examples: Training examples
            max_steps: Maximum steps in trajectory
            
        Returns:
            Collected trajectory
        """
        trajectory = Trajectory()
        current_program = copy.deepcopy(program)
        
        max_steps = max_steps or self.max_prompt_modifications
        
        for step in range(max_steps):
            # Current state (program configuration)
            state = self._get_state(current_program, examples[0] if examples else None)
            
            # Select action
            action = self._select_action(state)
            
            # Apply action to get new program
            new_program = self._modify_prompt(current_program, action)
            
            # Evaluate with TFLL metric
            reward = 0.0
            logprob = -1.0  # Default negative logprob
            
            with metric_only_mode():
                for example in examples[:5]:  # Sample a few examples
                    if self.metric:
                        score = self.metric(example, new_program)
                        reward += score
                        # Convert to logprob - for simple metrics use small value
                        logprob = max(-5.0, score / 10.0) if score < 0 else -0.1
                        
            reward /= min(5, len(examples))
            logprob /= min(5, len(examples))
            
            # Get baseline value
            value = self._get_baseline_value(state)
            
            # Add to trajectory
            trajectory.add(state, action, reward, logprob, value)
            
            # Update program for next step
            current_program = new_program
            
        return trajectory
    
    def _get_state(self, program: Module, example: Optional[Example]) -> Dict[str, Any]:
        """Extract state representation from program and example."""
        state = {
            "instructions": "",
            "num_instructions": 0,
            "has_cot": False,
            "has_step_by_step": False,
        }
        
        if hasattr(program, "signature") and hasattr(program.signature, "instructions"):
            instructions = program.signature.instructions or ""
            state["instructions"] = instructions
            # Count non-empty sentences
            sentences = [s.strip() for s in instructions.split(".") if s.strip()]
            state["num_instructions"] = len(sentences)
            state["has_cot"] = "chain of thought" in instructions.lower()
            state["has_step_by_step"] = "step by step" in instructions.lower()
            
        return state
    
    def _select_action(self, state: Dict[str, Any]) -> str:
        """Select action based on current state."""
        # Simple epsilon-greedy or softmax policy
        if self.rng.random() < 0.1:  # Exploration
            return self.rng.choice(self.action_space)
        else:
            # Choose based on state (simple heuristic for now)
            if not state["has_cot"] and state["num_instructions"] < 3:
                return "add_chain_of_thought"
            elif not state["has_step_by_step"] and state["num_instructions"] < 4:
                return "add_step_by_step"
            elif state["num_instructions"] > 5:
                return "simplify_instruction"
            else:
                return self.rng.choice(self.action_space)
    
    def _get_baseline_value(self, state: Dict[str, Any]) -> float:
        """Get baseline value for variance reduction."""
        if self.baseline_type == "mean":
            return self.baseline_value
        elif self.baseline_type == "moving_average":
            return self.baseline_value
        else:  # value_network would go here
            return 0.0
    
    def _update_baseline(self, returns: np.ndarray):
        """Update baseline based on observed returns."""
        if self.baseline_type == "mean":
            self.baseline_value = np.mean(returns)
        elif self.baseline_type == "moving_average":
            self.baseline_value = (
                self.baseline_alpha * np.mean(returns) + 
                (1 - self.baseline_alpha) * self.baseline_value
            )
    
    def _policy_update(
        self, 
        trajectories: List[Trajectory],
        program: Module
    ) -> Module:
        """
        Update policy based on collected trajectories.
        
        Args:
            trajectories: List of collected trajectories
            program: Current program
            
        Returns:
            Updated program
        """
        # Start with current program
        best_program = copy.deepcopy(program)
        accepted_modifications = []
        
        for trajectory in trajectories:
            # Compute advantages
            advantages = trajectory.compute_advantages(self.gamma, self.lam)
            
            # Update baseline
            returns = trajectory.compute_returns(self.gamma)
            self._update_baseline(returns)
            
            # Try modifications from trajectory
            for t in range(len(trajectory.actions)):
                # Only consider if this step had positive advantage
                if advantages[t] > self.advantage_threshold:
                    action = trajectory.actions[t]
                    
                    # Apply modification
                    modified = self._modify_prompt(copy.deepcopy(program), action)
                    
                    # Evaluate modified program using advantage-weighted score
                    # This is the policy gradient signal: advantage * log_prob
                    policy_gradient_score = advantages[t] * trajectory.logprobs[t]
                    
                    # Accept based on policy gradient signal
                    if policy_gradient_score > 0:
                        logger.info(f"Accept '{action}': PG={policy_gradient_score:.4f}")
                        best_program = copy.deepcopy(modified)
                        accepted_modifications.append((action, policy_gradient_score))
                    else:
                        logger.debug(f"Reject '{action}': PG={policy_gradient_score:.4f}")
        
        if not accepted_modifications:
            logger.info("No modifications accepted, keeping current program")
        else:
            logger.info(f"Accepted {len(accepted_modifications)} modifications")
                        
        return best_program
    
    def _policy_update_with_buffer(self, trajectories, program, n_recent=10):
        """Use N most recent buffer experiences."""
        if not self.replay_buffer or len(self.replay_buffer) < 5:
            return self._policy_update(trajectories, program)
        
        # Sample and add recent experiences
        batch = self.replay_buffer.sample_batch(n_recent)
        for exp in batch:
            if exp['advantage'] > 0:
                t = Trajectory()
                t.add(exp['state'], exp['action'], 
                      exp['reward'], exp['logprob'], 0)
                trajectories.append(t)
        return self._policy_update(trajectories, program)
    
    def _evaluate_program_score(self, program, trajectories):
        """Evaluate program score from trajectory data."""
        if not trajectories or not trajectories[0].rewards:
            return float("-inf")
    
        # Average reward across all trajectories  
        all_rewards = []
        for traj in trajectories:
            if traj.rewards:
                all_rewards.extend(traj.rewards)
        
        return np.mean(all_rewards) if all_rewards else float("-inf")
    
    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        valset: Optional[List[Example]] = None,
        **kwargs
    ) -> Module:
        """
        Optimize the student program using TFLL-based RL.
        
        Args:
            student: The student program to optimize
            trainset: Training examples
            teacher: Optional teacher program (unused)
            valset: Optional validation set
            
        Returns:
            Optimized student program
        """
        logger.info(f"Starting TFLL RL optimization with {self.num_updates} updates")
        
        if not self.metric:
            raise ValueError("TFLL metric must be provided")
            
        current_program = copy.deepcopy(student)
        
        # Track recent experiences for prompt updater
        recent_experiences = []
        
        for update in range(self.num_updates):
            # Single-step update
            # Sample just 1-2 examples for efficiency
            batch_size = min(2, len(trainset))
            batch = self.rng.sample(trainset, batch_size)
            
            # Get current state and select action
            state = self._get_state(current_program, batch[0] if batch else None)
            action = self._select_action(state)
            
            # Try modification
            new_program = self._modify_prompt(current_program, action)
            
            # Evaluate improvement using logprobs
            old_logprob = 0.0
            new_logprob = 0.0
            with metric_only_mode():
                for ex in batch:
                    if self.metric:
                        # Get average token logprobs
                        old_logprob += self.metric(ex, current_program)
                        new_logprob += self.metric(ex, new_program)
            old_logprob /= len(batch)
            new_logprob /= len(batch)
            
            # Compute reward and policy gradient
            reward = new_logprob - old_logprob
            advantage = reward - self.baseline_value
            policy_gradient = new_logprob * advantage
            
            # Accept based on policy gradient
            if policy_gradient > 0:
                logger.info(f"Accept '{action}': PG={policy_gradient:.4f}, reward={reward:.4f}")
                current_program = new_program
                # Update baseline
                self.baseline_value = (1 - self.baseline_alpha) * self.baseline_value + self.baseline_alpha * reward
            
            # Store experience with policy gradient
            recent_experiences.append((state, action, reward, policy_gradient))
            if len(recent_experiences) > 10:
                recent_experiences.pop(0)
            
            # Add to replay buffer if enabled
            if self.replay_buffer:
                self.replay_buffer.add(state, action, reward, new_logprob, advantage)
            
            # Log buffer statistics
            if self.replay_buffer and update % 5 == 0:
                logger.info(f"Buffer size: {len(self.replay_buffer)}, Total added: {self.replay_buffer.total_added}")
            
            # Evaluate on validation set
            if valset and update % 10 == 0:
                with metric_only_mode():
                    val_score = 0.0
                    for example in valset[:10]:
                        val_score += self.metric(example, current_program)
                    val_score /= min(10, len(valset))
                    logger.info(f"Update {update}: Validation score = {val_score:.4f}")
                    
        logger.info("TFLL RL optimization complete")
        return current_program
    
    def make_env(self, program: Module, trainset: List[Example]):
        """
        Create a Gym-compatible environment for the DSPy program.
        
        Args:
            program: DSPy program to optimize
            trainset: Training examples
            
        Returns:
            DSPyEnvironment instance
        """
        if self.use_gym_interface:
            from dspy.teleprompt.tfll_rl_env import DSPyEnvironment
            return DSPyEnvironment(
                program=program,
                trainset=trainset,
                metric=self.metric,
                action_space=self.action_space,
                modify_prompt_fn=self._modify_prompt,
                seed=self.rng.randint(0, 2**32 - 1)
            )
        else:
            raise ValueError("Gym interface not enabled. Set use_gym_interface=True")