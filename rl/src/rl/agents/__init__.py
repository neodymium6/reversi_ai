from abc import ABC, abstractmethod
from pprint import pprint
import random
from typing import List, Tuple, TypedDict
import numpy as np
from rust_reversi import AlphaBetaSearch, Board, PieceEvaluator, Turn, WinrateEvaluator, ThunderSearch
import torch
import matplotlib.pyplot as plt
import tqdm
from rl.memory import Memory, MemoryConfig, MemoryType
from rl.memory.proportional import ProportionalMemory
from rl.memory.simple import SimpleMemory
from rl.agents.batch_board import BatchBoard

class AgentConfig(TypedDict):
    memory_config: MemoryConfig
    batch_size: int
    board_batch_size: int
    device: torch.device
    eps_start: float
    eps_end: float
    eps_decay: float
    lr: float
    gradient_clip: float
    gamma: float
    n_episodes: int
    episodes_per_optimize: int
    episodes_per_target_update: int
    model_path: str
    verbose: bool

class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.net: torch.nn.Module = None
        self.target_net: torch.nn.Module = None
        self.config = config
        self.optimizer: torch.optim.Optimizer = None
        self.criterion: torch.nn.Module = None
        self.losses = []
        self.memory: Memory = None
        if config["memory_config"]["memory_type"] == MemoryType.UNIFORM:
            self.memory = SimpleMemory(config["memory_config"]["memory_size"])
        elif config["memory_config"]["memory_type"] == MemoryType.PROPORTIONAL:
            self.memory = ProportionalMemory(config["memory_config"]["memory_size"])
        else:
            raise ValueError("Invalid memory type")

    def after_init(self):
        self.target_net.load_state_dict(self.net.state_dict())
        self.net.to(self.config["device"])
        self.target_net.to(self.config["device"])
        if self.config["verbose"]:
            for param in self.net.parameters():
                print(f"Device: {param.device}")
                break
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.config["lr"])
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["lr"] * 10,
            # multiply by 1.1 because of increase of states by pass action (game may not end in 60 moves)
            total_steps=int(1.1 * self.config["n_episodes"]) // self.config["episodes_per_optimize"],
        )
        self.criterion = torch.nn.SmoothL1Loss()
        if self.config["verbose"]:
            pprint(self.config)

    @abstractmethod
    def get_action(self, board: Board, episoide: int) -> int:
        pass

    @abstractmethod
    def get_action_batch(self, boards: List[Board], episoide: int) -> List[int]:
        pass

    @abstractmethod
    def board_to_input(self, board: Board) -> torch.Tensor:
        pass

    @abstractmethod
    def update_target_net(self):
        pass

    def get_epsilon(self, episode: int) -> float:
        return self.config["eps_start"] + (self.config["eps_end"] - self.config["eps_start"]) * (1 - np.exp(-episode * self.config["eps_decay"] / 1000))

    def optimize(self) -> float:
        if len(self.memory) < self.config["batch_size"]:
            return 0.0
        self.net.train()
        self.target_net.eval()
        batch, indices = self.memory.sample(self.config["batch_size"])
        states, actions, next_states, rewards = zip(*batch)
        next_states: Tuple[Board, ...] = next_states
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
            # 64th element is pass
            # pass is only legal when the player has no legal moves
            legal_actions: torch.Tensor = torch.tensor(
                [ns.get_legal_moves_tf() + [ns.is_pass()] for ns in next_states],
                dtype=torch.bool,
                device=self.config["device"]
            )
            q_ns = q_ns.masked_fill(~legal_actions, -1e9)
            v_ns: torch.Tensor = q_ns.max(1).values
            # The value of the next state is the value of the opponent (1 - value is the value of the player)
            v_ns = 1.0 - v_ns
            game_overs = torch.tensor([ns.is_game_over() for ns in next_states], dtype=torch.bool, device=self.config["device"])
            v_ns = v_ns.masked_fill(game_overs, 0.0)    # If the game is over, the value of the next state is 0 and the reward is the final reward
        target = rewards + self.config["gamma"] * v_ns

        if isinstance(self.memory, ProportionalMemory):
            diff = (q_s_a - target).abs().detach().cpu().numpy().tolist()
            self.memory.update_priorities(indices, diff)

        loss: torch.Tensor = self.criterion(q_s_a, target)
        loss_value = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), self.config["gradient_clip"])
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss_value

    def train(self):
        iter_size = self.config["n_episodes"] // self.config["board_batch_size"]
        optimize_count = 0
        step_count = 0
        if self.config["verbose"]:
            episodes_iter = tqdm.tqdm(range(iter_size))
        else:
            episodes_iter = range(iter_size)
        for i in episodes_iter:
            board_batch = BatchBoard(self.config["board_batch_size"])

            while not board_batch.is_game_over():
                states = board_batch.get_boards()
                actions = self.get_action_batch(states, i * self.config["board_batch_size"])
                next_states, rewards = board_batch.do_move(actions)

                for state, action, next_state, reward in zip(states, actions, next_states, rewards):
                    self.memory.push(state, action, next_state, reward)
                step_count += 1

                scale = 100
                memory_size_per_episode = 60
                if scale * step_count % (scale * memory_size_per_episode * self.config["episodes_per_optimize"] // self.config["board_batch_size"]) == 0:
                    loss = self.optimize()
                    optimize_count += 1
                    self.losses.append((optimize_count, loss))

                if scale * step_count % (scale * memory_size_per_episode * self.config["episodes_per_target_update"] // self.config["board_batch_size"]) == 0:
                    self.update_target_net()

            if i % (iter_size // 10) == 0:
                if self.config["verbose"]:
                    win_rate = self.vs_random(1000)
                    print(f"Episode {i * self.config['board_batch_size']}: Win rate vs random = {win_rate}")
                if i != 0:
                    self.save()
                    self.plot()
        if self.config["verbose"]:
            print("Training finished")
            win_rate = self.vs_random(1000)
            print(f"Win rate vs random = {win_rate}")
        self.save()
        self.plot()

    def save(self):
        if self.config["verbose"]:
            print(f"Saving model to {self.config['model_path']}")
        torch.save(self.net.state_dict(), self.config["model_path"])

    def plot(self):
        def calculate_moving_average(data, window_size):
            results = []
            for i in range(len(data)):
                if i < window_size - 1:
                    results.append(None)
                else:
                    window = data[i-window_size+1:i+1]
                    window_average = sum(window) / window_size
                    results.append(window_average)
            return results
        fig, ax = plt.subplots()
        (optimize_count, loss) = zip(*self.losses)
        ma100 = calculate_moving_average(loss, 100)
        ax.plot(optimize_count, ma100, label="Moving average loss (window=100)")
        ma1000 = calculate_moving_average(loss, 1000)
        ax.plot(optimize_count, ma1000, label="Moving average loss (window=1000)")
        ax.set_xlabel("Optimize count")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        fig.legend()
        plt.savefig("loss.png")

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, weights_only=True))

    def vs_random(self, n_games: int) -> float:
        if self.config["verbose"]:
            print("Vs Random")
        self.net.eval()
        def two_game():
            win_count = 0
            # agent is black
            board = Board()
            while not board.is_game_over():
                if board.is_pass():
                    board.do_pass()
                    continue
                if board.get_turn() == Turn.BLACK:
                    action = self.get_action(board, 1 << 10)
                else:
                    action = board.get_random_move()
                board.do_move(action)
            if board.is_black_win():
                win_count += 1
            # agent is white
            board = Board()
            while not board.is_game_over():
                if board.is_pass():
                    board.do_pass()
                    continue
                if board.get_turn() == Turn.WHITE:
                    action = self.get_action(board, 1 << 10)
                else:
                    action = board.get_random_move()
                board.do_move(action)
            if board.is_white_win():
                win_count += 1
            return win_count
    
        win_count = 0
        for _ in range(n_games // 2):
            win_count += two_game()
        win_rate = win_count / n_games
        return win_rate

    def vs_alpha_beta(self, n_games: int, epsilon: float = 0.1) -> float:
        if self.config["verbose"]:
            print("Vs AlphaBeta")
        self.net.eval()
        def two_game():
            evaluator = PieceEvaluator()
            search = AlphaBetaSearch(evaluator, 3, 1 << 10)
            win_count = 0
            # agent is black
            board = Board()
            while not board.is_game_over():
                if board.is_pass():
                    board.do_pass()
                    continue
                if random.random() < epsilon:
                    action = board.get_random_move()
                    board.do_move(action)
                    continue
                if board.get_turn() == Turn.BLACK:
                    action = self.get_action(board, 1 << 10)
                else:
                    action = search.get_move(board)
                board.do_move(action)
            if board.is_black_win():
                win_count += 1
            # agent is white
            board = Board()
            while not board.is_game_over():
                if board.is_pass():
                    board.do_pass()
                    continue
                if random.random() < epsilon:
                    action = board.get_random_move()
                    board.do_move(action)
                    continue
                if board.get_turn() == Turn.WHITE:
                    action = self.get_action(board, 1 << 10)
                else:
                    action = search.get_move(board)
                board.do_move(action)
            if board.is_white_win():
                win_count += 1
            return win_count
    
        win_count = 0
        for _ in range(n_games // 2):
            win_count += two_game()
        win_rate = win_count / n_games
        return win_rate
    
    def evaluate(self, board: Board) -> float:
        self.net.eval()
        board_tensor = self.board_to_input(board)
        board_tensor = board_tensor.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net(board_tensor)
        legal_actions: torch.Tensor = torch.tensor(
            board.get_legal_moves_tf() + [board.is_pass()],
            dtype=torch.bool,
            device=self.config["device"]
        )
        out = out.masked_fill(~legal_actions, -1e9)
        out = out.cpu().numpy()
        score = out.max()
        return np.clip(score, 0.0, 1.0)
    
    def thunder_vs_alpha_beta(self, n_games: int, epsilon: float = 0.1) -> float:
        if self.config["verbose"]:
            print("Thunder vs AlphaBeta")
        class NetEvaluator(WinrateEvaluator):
            def __init__(self):
                self.agent = None
                super().set_py_evaluator(self)
            def set_agent(self, agent):
                self.agent = agent
            def evaluate(self, board: Board) -> float:
                return self.agent.evaluate(board)
        self.net.eval()
        net_evaluator = NetEvaluator()
        net_evaluator.set_agent(self)
        thunder_search = ThunderSearch(net_evaluator, 100, 0.01)
        piece_evaluator = PieceEvaluator()
        alpha_beta_search = AlphaBetaSearch(piece_evaluator, 3, 1 << 10)
        def two_game():
            win_count = 0
            # agent is black
            board = Board()
            while not board.is_game_over():
                if board.is_pass():
                    board.do_pass()
                    continue
                if random.random() < epsilon:
                    action = board.get_random_move()
                    board.do_move(action)
                    continue
                if board.get_turn() == Turn.BLACK:
                    action = thunder_search.get_move(board)
                else:
                    action = alpha_beta_search.get_move(board)
                board.do_move(action)
            if board.is_black_win():
                win_count += 1
            # agent is white
            board = Board()
            while not board.is_game_over():
                if board.is_pass():
                    board.do_pass()
                    continue
                if random.random() < epsilon:
                    action = board.get_random_move()
                    board.do_move(action)
                    continue
                if board.get_turn() == Turn.WHITE:
                    action = thunder_search.get_move(board)
                else:
                    action = alpha_beta_search.get_move(board)
                board.do_move(action)
            if board.is_white_win():
                win_count += 1
            return win_count
        
        win_count = 0
        if self.config["verbose"]:
            games_iter = tqdm.trange(n_games // 2)
        else:
            games_iter = range(n_games // 2)
        for _ in games_iter:
            win_count += two_game()
        win_rate = win_count / n_games
        return win_rate
