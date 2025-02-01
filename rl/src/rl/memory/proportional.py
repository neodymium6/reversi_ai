from typing import List, Tuple
from rust_reversi import Board
from rl.memory import Memory
import numpy as np

EPSILON = 1e-6

class ProportionalMemory(Memory):
    def __init__(self, maxlen):
        self.capacity = maxlen
        self.memory = []
        self.priorities = []
        self.max_priority = 1.0
        self.position = 0

    def push(self, state: Board, action: int, next_state: Board, reward: float) -> None:
        max_priority = max(self.priorities) if self.priorities else self.max_priority

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, next_state, reward))
            self.priorities.append(max_priority)
        else:
            # if memory is full, replace the oldest memory
            self.memory[self.position] = (state, action, next_state, reward)
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> Tuple[List[Tuple[Board, int, Board, float]], List[int]]:
        if len(self.memory) < batch_size:
            indices = list(range(len(self.memory)))
            return self.memory, indices
        
        properties = np.array(self.priorities)
        probs = properties / np.sum(properties)

        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)

        return [self.memory[i] for i in indices], indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        for i, priority in zip(indices, priorities):
            self.priorities[i] = max(priority, EPSILON)
            self.max_priority = max(self.max_priority, self.priorities[i])

    def __len__(self) -> int:
        return len(self.memory)
