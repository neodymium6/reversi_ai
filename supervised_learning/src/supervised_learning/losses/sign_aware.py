import torch
import torch.nn as nn

class SignAwareMAE(nn.Module):
    def __init__(self, sign_penalty_weight=2.0):
        super().__init__()
        self.sign_penalty_weight = sign_penalty_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base_loss: torch.Tensor = torch.abs(pred - target)

        sign_mismatch = (pred * target < 0).float()

        sign_penalty = sign_mismatch * torch.abs(target)

        total_loss = base_loss + self.sign_penalty_weight * sign_penalty
        
        return total_loss.mean()
