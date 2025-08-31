"""DSPy module for experience-based prompt updates."""

import dspy
from typing import List, Dict, Any, Tuple


class ExperiencePromptUpdater(dspy.Module):
    """Analyzes experiences to generate prompt updates."""
    
    def __init__(self):
        super().__init__()
        
        # Signature for analyzing experiences
        self.signature = dspy.Signature(
            "experiences -> search, replace"
        )
        
        self.predictor = dspy.Predict(self.signature)
    
    def forward(self, experiences):
        """Generate search/replace from experiences."""
        exp_text = []
        for obs, action, reward in experiences:
            exp_text.append(f"Action: {action}, Reward: {reward:.2f}")
        
        result = self.predictor(experiences="\n".join(exp_text))
        return result.search, result.replace


class RuleBasedUpdater(dspy.Module):
    """Simple rule-based prompt updater."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, current_prompt, experiences):
        """Find best action and return search/replace."""
        if not experiences:
            return current_prompt, current_prompt
        
        # Find action with highest reward
        best = max(experiences, key=lambda x: x[2])
        obs, action, reward = best
        
        if reward > 0:
            suffix = "\nThink step-by-step."
            return current_prompt, current_prompt + suffix
        
        return current_prompt, current_prompt