from abc import abstractmethod
import random
from rust_reversi import Board
import torch
from rl.agents import Agent, AgentConfig

class QnetAgent(Agent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.net: torch.nn.Module = None
        self.target_net: torch.nn.Module = None

    @abstractmethod
    def board_to_input(self, board: Board) -> torch.Tensor:
        pass

    @abstractmethod
    def update_target_net(self):
        pass

    def get_action(self, board: Board, episode: int) -> int:
        if board.is_pass():
            return 64
        epsilon = self.get_epsilon(episode)
        if random.random() < epsilon:
            return random.choice(board.get_legal_moves_vec())
        self.net.eval()
        board_tensor = self.board_to_input(board)
        board_tensor = board_tensor.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensor)
        # 64th element is pass
        legal_actions: torch.Tensor = torch.tensor(
            board.get_legal_moves_tf() + [board.is_pass()],
            dtype=torch.bool,
            device=self.config["device"],
        )
        out = out.masked_fill(~legal_actions, -1e9)
        return out.argmax().item()
    
    def get_action_batch(self, boards: list[Board], episoide: int) -> list[int]:
        self.net.eval()
        board_tensors = torch.stack([self.board_to_input(board) for board in boards])
        board_tensors = board_tensors.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensors)
        # 64th element is pass
        legal_actions = torch.stack([torch.tensor(
                board.get_legal_moves_tf() + [board.is_pass()],
                dtype=torch.bool,
                device=self.config["device"]
        ) for board in boards])
        out = out.masked_fill(~legal_actions, -1e9)
        actions = out.argmax(dim=1).tolist()
        # override with epsilon greedy
        epsilon = self.get_epsilon(episoide)
        for i, board in enumerate(boards):
            if board.is_pass():
                actions[i] = 64
                continue
            if random.random() < epsilon:
                actions[i] = random.choice(board.get_legal_moves_vec())
        return actions
