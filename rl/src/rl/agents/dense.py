from rl.agents import Agent
from rl.memory.simple import SimpleMemory
from rust_reversi import Board
from rl.models.dense import DenseNet
from typing import List, TypedDict
import torch
import torchinfo
import random
import tqdm

class DenseAgentConfig(TypedDict):
    memory_size: int
    hidden_size: int
    batch_size: int
    device: torch.device
    eps_start: float
    eps_end: float
    lr: float
    gamma: float
    n_episodes: int
    verbose: bool


class DenseAgent(Agent):
    def __init__(self, config: DenseAgentConfig):
        super().__init__()
        self.memory = SimpleMemory(config["memory_size"])
        self.net = DenseNet(128, config["hidden_size"], 64)
        self.target_net = DenseNet(128, config["hidden_size"], 64)
        self.target_net.load_state_dict(self.net.state_dict())
        self.net.to(config["device"])
        self.target_net.to(config["device"])
        if config["verbose"]:
            torchinfo.summary(self.net, input_size=(config["batch_size"], 128), device=config["device"])
            for param in self.net.parameters():
                print(f"Device: {param.device}")
                break
        self.config = config
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config["lr"])
        self.criterion = torch.nn.SmoothL1Loss()

    def board_to_input(self, board: Board) -> torch.Tensor:
        res = torch.zeros(128, dtype=torch.float32)
        player_board, opponent_board, _turn = board.get_board()
        for i in range(64):
            bit = 1 << (64 - i - 1)
            if player_board & bit:
                res[i] = 1.0
            if opponent_board & bit:
                res[i + 64] = 1.0
        return res

    def get_action(self, board: Board, progress: float) -> int:
        epsilon = self.config["eps_start"] + (self.config["eps_end"] - self.config["eps_start"]) * progress
        if random.random() < epsilon:
            return random.choice(board.get_legal_moves_vec())
        self.net.eval()
        board_tensor = self.board_to_input(board)
        board_tensor = board_tensor.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensor)
        legal_actions: List[bool] = board.get_legal_moves_tf()
        legal_actions = [1 if x else 0 for x in legal_actions]
        out = out.cpu().numpy()
        out = out * legal_actions
        return out.argmax()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def optimize(self):
        if len(self.memory) < self.config["batch_size"]:
            return
        self.net.train()
        self.target_net.eval()
        batch = self.memory.sample(self.config["batch_size"])
        states, actions, next_states, rewards = zip(*batch)
        states = torch.stack([self.board_to_input(x) for x in states])
        states = states.to(self.config["device"])
        next_states_t = torch.stack([self.board_to_input(x) for x in next_states])
        next_states_t = next_states_t.to(self.config["device"])
        actions = torch.tensor(actions, dtype=torch.int64, device=self.config["device"])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.config["device"])
        q_s: torch.Tensor = self.net(states)
        q_s_a = q_s.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_ns: torch.Tensor = self.target_net(next_states_t)
            v_ns: torch.Tensor = q_ns.max(1).values
        for ns_idx, ns in enumerate(next_states):
            if ns.is_game_over():
                v_ns[ns_idx] = 0.0      # If the game is over, the value of the next state is 0 and the reward is the final reward
        target = rewards + self.config["gamma"] * v_ns
        loss: torch.Tensor = self.criterion(q_s_a, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self):
        if self.config["verbose"]:
            episodes_iter = tqdm.tqdm(range(self.config["n_episodes"]))
        else:
            episodes_iter = range(self.config["n_episodes"])
        for i in episodes_iter:
            board = Board()

            while True:
                if board.is_pass():
                    board.do_pass()
                action = self.get_action(board, i / self.config["n_episodes"])
                next_board = board.clone()
                next_board.do_move(action)

                if next_board.is_game_over():
                    if next_board.is_win():
                        reward = 0.0        # turn swapped in do_move, so is_win menas the player that just moved lost
                    elif next_board.is_lose():
                        reward = 1.0        # turn swapped in do_move, so is_lose menas the player that just moved won
                    else:
                        reward = 0.5        # draw
                    
                    self.memory.push(board, action, next_board, reward)
                    break
                else:
                    self.memory.push(board, action, next_board, 0.0)
                    board = next_board
                self.optimize()
            self.update_target_net()

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, weights_only=True))