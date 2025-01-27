from rl.agents import Agent
from rl.memory.simple import SimpleMemory

class DenseAgent(Agent):
    def __init__(self):
        super().__init__()
        self.memory = SimpleMemory()
    pass
