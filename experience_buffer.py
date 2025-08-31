"""Experience buffer for TFLL RL optimizer."""
from collections import deque
import random
from typing import List, Dict, Any


class ExperienceBuffer:
    """Buffer for storing and reusing trajectory experiences."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize experience buffer."""
        self.buffer = deque(maxlen=max_size)
        self.total_added = 0