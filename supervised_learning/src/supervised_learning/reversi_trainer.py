import torch
from typing import Iterable, Callable, Tuple
import tqdm
from supervised_learning.models import ReversiNet
import matplotlib.pyplot as plt
from supervised_learning.vs import vs_random, vs_mcts, vs_alpha_beta
import time

class Trainer:
    def __init__(
            self,
            net: ReversiNet,
            get_data_loader_func: Callable[[bool], Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]],
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            device: torch.device,
            loss_plot_path: str,
            model_path: str,
            epochs: int,
            max_lr: float,
            middle_n_games: int = 100,
            final_n_games: int = 500,
            data_loader_update_per_epoch: int = 10,
            verbose: bool = True,
        ):
        self.net = net
        self.train_loader, self.test_loader, self.get_dataloader_time = get_data_loader_func(verbose)
        self.get_data_loader_func = get_data_loader_func
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.loss_plot_path = loss_plot_path
        self.model_path = model_path
        self.epochs = epochs
        self.middle_n_games = middle_n_games
        self.final_n_games = final_n_games
        self.data_loader_update_per_epoch = data_loader_update_per_epoch
        self.verbose = verbose
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(self.train_loader),
        )
        self.train_losses = []
        self.test_losses = []
        self.lrs = []

    def plot_loss(self, epoch: int) -> None:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color1 = 'blue'
        color2 = 'orange'
        ax1.plot(self.train_losses, label="train", color=color1)
        ax1.plot(self.test_losses, label="test", color=color2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("log")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color3 = 'red'
        ax2.plot(self.lrs, label="learning rate", color=color3, linestyle='--')
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")
        ax2.legend(loc='upper right')
        
        plt.title(f"Training Progress - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(self.loss_plot_path)
        plt.close(fig)

    def eval_with_win_rate(self, n_games) -> str:
        random_win_rate = vs_random(n_games, self.net)
        mcts_win_rate = vs_mcts(n_games, self.net)
        alpha_beta_win_rate = vs_alpha_beta(n_games, self.net)
        return f"Win rate vs random: {random_win_rate:.4f}, vs MCTS: {mcts_win_rate:.4f}, vs alpha beta: {alpha_beta_win_rate:.4f}"

    def train(self) -> Iterable[None]:
        lr_scheduler_step = 0
        lr_scheduler_step_max = self.epochs * len(self.train_loader)
        epoch_pb = tqdm.tqdm(range(self.epochs), leave=False)
        for epoch in epoch_pb:
            self.net.train()
            total_loss = 0.0
            train_pb = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
            for inputs, targets in train_pb:
                inputs: torch.Tensor
                targets: torch.Tensor
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs: torch.Tensor = self.net(inputs)
                loss: torch.Tensor = self.criterion(outputs.view(-1), targets)
                total_loss += loss.item()
                train_pb.set_description(f"Epoch {epoch} - Loss: {loss.item():5.2f}")
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                loss.backward()
                self.optimizer.step()
                if lr_scheduler_step < lr_scheduler_step_max:
                    # iterative dataset with workers may cause more steps than total_steps
                    lr_scheduler_step += 1
                    self.lr_scheduler.step()
            self.train_losses.append(total_loss / len(self.train_loader))
            self.net.eval()
            with torch.no_grad():
                total_loss = 0.0
                test_pb = tqdm.tqdm(self.test_loader, desc=f"Epoch {epoch} - Test", leave=False)
                for inputs, targets in test_pb:
                    inputs: torch.Tensor
                    targets: torch.Tensor
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs: torch.Tensor = self.net(inputs)
                    loss = self.criterion(outputs.view(-1), targets)
                    total_loss += loss.item()
                    test_pb.set_description(f"Epoch {epoch} - Test Loss: {loss.item():5.2f}")
                self.test_losses.append(total_loss / len(self.test_loader))
                self.lrs.append(self.lr_scheduler.get_last_lr()[0])
            self.plot_loss(epoch)

            if epoch % (self.epochs // 10) == 0:
                torch.save(self.net.state_dict(), self.model_path)
                if self.verbose:
                    epoch_pb.write(f"Epoch {epoch:{len(str(self.epochs))}d}: {self.eval_with_win_rate(self.middle_n_games)}, Loss: {self.train_losses[-1]:5.2f}, Test Loss: {self.test_losses[-1]:5.2f}")

            if epoch % self.data_loader_update_per_epoch == self.data_loader_update_per_epoch - 1 and epoch != self.epochs - 1:
                start_time = time.time()
                end_time_pred = time.localtime(start_time + self.get_dataloader_time)
                end_time_pred = time.strftime("%H:%M:%S", end_time_pred)
                if self.verbose:
                    epoch_pb.write(f"Updating data loaders... Expected to finish at {end_time_pred}")
                self.train_loader, self.test_loader, self.get_dataloader_time = self.get_data_loader_func(False)
            yield
        torch.save(self.net.state_dict(), self.model_path)
        self.plot_loss(self.epochs)
        if self.verbose:
            epoch_pb.write(f"Epoch {self.epochs:{len(str(self.epochs))}d}: {self.eval_with_win_rate(self.final_n_games)}, Loss: {self.train_losses[-1]:5.2f}, Test Loss: {self.test_losses[-1]:5.2f}")
