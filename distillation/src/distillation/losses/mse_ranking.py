import torch

class MSEWithRankingLoss(torch.nn.Module):
    def __init__(self, rank_weight: float = 0.1, margin: float = 0.1):
        super(MSEWithRankingLoss, self).__init__()
        self.rank_weight = rank_weight
        self.margin = margin
        self.mse = torch.nn.MSELoss()

    def _compute_ranking_loss(self, student_q: torch.Tensor, teacher_q: torch.Tensor) -> torch.Tensor:
        teacher_advantages = teacher_q.unsqueeze(1) - teacher_q.unsqueeze(2)
        student_advantages = student_q.unsqueeze(1) - student_q.unsqueeze(2)

        ranking_loss = torch.mean(
            torch.relu(
                -(teacher_advantages * student_advantages) + self.margin
            )
        )

        return ranking_loss
    
    def forward(self, student_q: torch.Tensor, teacher_q: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(student_q, teacher_q)
        ranking_loss = self._compute_ranking_loss(student_q, teacher_q)
        return mse_loss + self.rank_weight * ranking_loss
