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
        self.max_p = 0.0

    def add(self, priority: float) -> None:
        # if the memory is full, replace the oldest memory
        self.update(self.pointer, priority)
        self.pointer = (self.pointer + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        # update max priority
        self.max_p = max(priority, self.max_p)
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

    def sample_indices(self, batch_size: int) -> List[int]:
        indices = []
        for _ in range(batch_size):
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
        return indices
    
    def sum(self) -> float:
        return self.tree[0]
    
    def __len__(self) -> int:
        return self.length
    
    def max(self) -> float:
        return self.max_p

class ProportionalMemory(Memory):
    def __init__(self, maxlen: int):
        is_power_of_two = lambda n: (n != 0) and (n & (n - 1) == 0)
        if not is_power_of_two(maxlen):
            maxlen = 2 ** (maxlen.bit_length())
            print(f"Warning: maxlen is not a power of 2, rounding up to the next power of 2: {maxlen}")
        self.capacity = maxlen
        self.tree = SumTree(maxlen)
        self.memory = [None] * maxlen

    def push(self, state: Board, action: int, next_state: Board, reward: float) -> None:
        max_priority = self.tree.max()
        self.memory[self.tree.pointer] = (state, action, next_state, reward)
        self.tree.add(max_priority)

    def sample(self, batch_size) -> Tuple[List[Tuple[Board, int, Board, float]], List[int]]:
        indices = self.tree.sample_indices(batch_size)
        return [self.memory[i] for i in indices], indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        priorities = np.array(priorities) + EPSILON
        for i, priority in zip(indices, priorities):
            self.tree.update(i, priority)

    def __len__(self) -> int:
        return len(self.tree)
