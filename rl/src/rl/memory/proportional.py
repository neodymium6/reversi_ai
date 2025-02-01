from typing import List, Tuple
from rust_reversi import Board
from rl.memory import Memory
import numpy as np

EPSILON = 1e-6

class SumTree:
    def __init__(self, capacity: int):
        # capacity must be a power of 2 to make the tree a complete binary tree
        is_power_of_two = lambda n: (n != 0) and (n & (n - 1) == 0)
        assert is_power_of_two(capacity), f"Capacity must be a power of 2, got {capacity}"
        
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.length = 0
        self.pointer = 0

    def add(self, priority: float) -> None:
        # if the memory is full, replace the oldest memory
        self.update(self.pointer, priority)
        self.pointer = (self.pointer + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        # idx to tree index
        idx = idx + self.capacity - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # get the parent of the node
        idx = (idx - 1) // 2
        # propagate the change through the tree (leaf to root)
        while idx >= 0:
            self.tree[idx] += change
            idx = (idx - 1) // 2

    def sample(self, batch_size: int) -> Tuple[List[int], List[float]]:
        indices = []
        priorities = []
        for i in range(batch_size):
            rand = np.random.uniform(0, self.tree[0])
            idx = 0
            while True:
                left = 2 * idx + 1
                right = left + 1
                if left >= len(self.tree):
                    break
                if rand <= self.tree[left]:
                    idx = left
                else:
                    rand -= self.tree[left]
                    idx = right
            # idx to memory index
            indices.append(idx - self.capacity + 1)
            priorities.append(self.tree[idx])
        return indices, priorities
    
    def sum(self) -> float:
        return self.tree[0]
    
    def __len__(self) -> int:
        return self.length

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
