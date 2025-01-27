import random
from typing import List, Tuple
from rust_reversi import Board
from rl.memory import Memory
from collections import deque

class SimpleMemory(Memory):
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def push(self, state: Board, action: int, next_state: Board, reward: float) -> None:
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size) -> List[Tuple[Board, int, Board, float]]:
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)
