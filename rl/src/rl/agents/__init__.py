from pprint import pprint
import random
from typing import Tuple, TypedDict
import numpy as np
from rust_reversi import AlphaBetaSearch, Board, Turn, WinrateEvaluator, ThunderSearch, MctsSearch, MatrixEvaluator
import torch
import matplotlib.pyplot as plt
import tqdm
from rl.memory import Memory, MemoryConfig, MemoryType
from rl.memory.proportional import ProportionalMemory
from rl.memory.simple import SimpleMemory
from rl.agents.batch_board import BatchBoard
from rl.agents.net_driver import NetDriver, NetConfig, NetType
from rl.agents.net_driver.dense import DenseDriver
from rl.agents.net_driver.cnn import CnnDriver

class AgentConfig(TypedDict):
    memory_config: MemoryConfig
    net_config: NetConfig
    batch_size: int
    board_batch_size: int
    n_board_init_random_moves: int
    p_board_init_random_moves: float
    device: torch.device
    eps_start: float
    eps_end: float
    eps_decay: float
    lr: float
    gradient_clip: float
    gamma: float
    n_episodes: int
    steps_per_optimize: int
    optimize_per_target_update: int
    verbose: bool
    model_path: str

class Agent():
    def __init__(self, config: AgentConfig):
        self.config = config
        self.losses = []
        self.memory: Memory = None
        self.net_driver: NetDriver = None
        if config["net_config"]["net_type"] == NetType.Dense:
            self.net_driver = DenseDriver(config["verbose"], config["device"], config["net_config"], config["batch_size"])
        elif config["net_config"]["net_type"] == NetType.Conv5 or config["net_config"]["net_type"] == NetType.Conv5Dueling or config["net_config"]["net_type"] == NetType.RESNET10 or config["net_config"]["net_type"] == NetType.Transformer:
            self.net_driver = CnnDriver(config["verbose"], config["device"], config["net_config"], config["batch_size"])
        else:
            raise ValueError("Invalid net type")

        if config["memory_config"]["memory_type"] == MemoryType.UNIFORM:
            self.memory = SimpleMemory(config["memory_config"]["memory_size"])
        elif config["memory_config"]["memory_type"] == MemoryType.PROPORTIONAL:
            self.memory = ProportionalMemory(config["memory_config"]["memory_size"], config["memory_config"]["alpha"], config["memory_config"]["beta"])
        else:
            raise ValueError("Invalid memory type")
        self.optimizer = torch.optim.AdamW(self.net_driver.net.parameters(), lr=config["lr"])
        self.criterion: torch.nn.Module = torch.nn.SmoothL1Loss(reduction="none")
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["lr"] * 10,
            # multiply by 1.1 because of increase of states by pass action (game may not end in 60 moves)
            total_steps=int(1.1 * self.config["n_episodes"] * 60 / (self.config["steps_per_optimize"] * self.config["board_batch_size"])),
            final_div_factor=1e7,
        )
        if self.config["verbose"]:
            pprint(self.config)

    def get_epsilon(self, episode: int) -> float:
        return self.config["eps_start"] + (self.config["eps_end"] - self.config["eps_start"]) * (1 - np.exp(-episode * self.config["eps_decay"] / 1000))

    def optimize(self) -> float:
        if len(self.memory) < self.config["batch_size"]:
            return 0.0
        self.net_driver.net.train()
        self.net_driver.target_net.eval()
        batch, indices, weights = self.memory.sample(self.config["batch_size"])
        states, actions, next_states, rewards = zip(*batch)
        next_states: Tuple[Board, ...] = next_states
        states = torch.stack([self.net_driver.board_to_input(x) for x in states])
        states = states.to(self.config["device"])
        next_states_t = torch.stack([self.net_driver.board_to_input(x) for x in next_states])
        next_states_t = next_states_t.to(self.config["device"])
        actions = torch.tensor(actions, dtype=torch.int64, device=self.config["device"])
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.config["device"])
        q_s: torch.Tensor = self.net_driver.net(states)
        q_s_a = q_s.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_ns: torch.Tensor = self.net_driver.target_net(next_states_t)
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
        loss = loss * torch.tensor(weights, dtype=torch.float32, device=self.config["device"])
        loss = loss.mean()
        loss_value = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net_driver.net.parameters(), self.config["gradient_clip"])
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss_value

    def train_iter(self, n_reports: int = 6):
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
                # Randomly initialize the board for the first few moves
                if board_batch.get_piece_mean() < self.config["n_board_init_random_moves"] and random.random() < self.config["p_board_init_random_moves"]:
                    board_batch.do_random_move()
                else:
                    states = board_batch.get_boards()
                    actions = self.net_driver.get_action_batch(states, self.get_epsilon(i * self.config["board_batch_size"]))
                    next_states, rewards = board_batch.do_move(actions)
                    for state, action, next_state, reward in zip(states, actions, next_states, rewards):
                        self.memory.push(state, action, next_state, reward)
                step_count += 1

                if step_count % self.config["steps_per_optimize"] == 0:
                    loss = self.optimize()
                    optimize_count += 1
                    self.losses.append((optimize_count, loss))

                if optimize_count % self.config["optimize_per_target_update"] == 0:
                    self.net_driver.update_target_net()

            if i != 0 and i % (iter_size // 5) == 0:
                self.save()
                self.plot()
            if i % (iter_size // (n_reports - 1)) == 0:
                win_rate1 = self.vs_random(1000)
                win_rate2 = self.vs_mcts(1000)
                win_rate3 = self.vs_alpha_beta(1000)
                metrics = {
                    "episode": i * self.config["board_batch_size"],
                    "vs_random": win_rate1,
                    "vs_mcts": win_rate2,
                    "vs_alpha_beta": win_rate3,
                }
                yield metrics
        win_rate1 = self.vs_random(1000)
        win_rate2 = self.vs_mcts(1000)
        win_rate3 = self.vs_alpha_beta(1000)
        metrics = {
            "episode": self.config["n_episodes"],
            "vs_random": win_rate1,
            "vs_mcts": win_rate2,
            "vs_alpha_beta": win_rate3,
        }
        yield metrics
        self.save()
        self.plot()

    def train(self):
        if self.config["verbose"]:
            print("Training started")
            for metrics in self.train_iter(n_reports=6):
                print(f"Episode {metrics['episode']}")
                print(f"Win rate vs Random: {metrics['vs_random']:.3f}, vs MCTS: {metrics['vs_mcts']:.3f}, vs AlphaBeta: {metrics['vs_alpha_beta']:.3f}")
            print("Training finished")
        else:
            for metrics in self.train_iter(n_reports=0):
                pass

    def save(self):
        self.net_driver.save(self.config["model_path"])

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
        plt.close(fig)

    def load(self, path: str):
        self.net_driver.load(path)

    def vs_random(self, n_games: int) -> float:
        if self.config["verbose"]:
            print("Vs Random")
        self.net_driver.net.eval()
        def two_game():
            win_count = 0
            # agent is black
            board = Board()
            while not board.is_game_over():
                if board.is_pass():
                    board.do_pass()
                    continue
                if board.get_turn() == Turn.BLACK:
                    action = self.net_driver.get_action(board, 0.0)
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
                    action = self.net_driver.get_action(board, 0.0)
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
        self.net_driver.net.eval()
        def two_game():
            evaluator = MatrixEvaluator([
                [ 40,   1,  4,  0,  0,  4,   1, 40],
                [  1, -12, -8, -6, -6, -8, -12,  1],
                [  4,  -8, -1,  0,  0, -1,  -8,  4],
                [  0,  -6,  0,  0,  0,  0,  -6,  0],
                [  0,  -6,  0,  0,  0,  0,  -6,  0],
                [  4,  -8, -1,  0,  0, -1,  -8,  4],
                [  1, -12, -8, -6, -6, -8, -12,  1],
                [ 40,   1,  4,  0,  0,  4,   1, 40],
            ])
            search = AlphaBetaSearch(evaluator, 4, 1 << 10)
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
                    action = self.net_driver.get_action(board, 0.0)
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
                    action = self.net_driver.get_action(board, 0.0)
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

    def vs_mcts(self, n_games: int, epsilon: float = 0.1) -> float:
        if self.config["verbose"]:
            print("Vs MCTS")
        self.net_driver.net.eval()
        def two_game():
            search = MctsSearch(100, 1.0, 3)
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
                    action = self.net_driver.get_action(board, 0.0)
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
                    action = self.net_driver.get_action(board, 0.0)
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
        self.net_driver.net.eval()
        board_tensor = self.net_driver.board_to_input(board)
        board_tensor = board_tensor.to(self.config["device"])
        with torch.no_grad():
            out: torch.Tensor = self.net_driver.net(board_tensor)
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
                self.agent: Agent = None
                super().set_py_evaluator(self)
            def set_agent(self, agent):
                self.agent = agent
            def evaluate(self, board: Board) -> float:
                return self.agent.evaluate(board)
        self.net_driver.net.eval()
        net_evaluator = NetEvaluator()
        net_evaluator.set_agent(self)
        thunder_search = ThunderSearch(net_evaluator, 100, 0.01)
        piece_evaluator = MatrixEvaluator([
            [ 40,   1,  4,  0,  0,  4,   1, 40],
            [  1, -12, -8, -6, -6, -8, -12,  1],
            [  4,  -8, -1,  0,  0, -1,  -8,  4],
            [  0,  -6,  0,  0,  0,  0,  -6,  0],
            [  0,  -6,  0,  0,  0,  0,  -6,  0],
            [  4,  -8, -1,  0,  0, -1,  -8,  4],
            [  1, -12, -8, -6, -6, -8, -12,  1],
            [ 40,   1,  4,  0,  0,  4,   1, 40],
        ])
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
