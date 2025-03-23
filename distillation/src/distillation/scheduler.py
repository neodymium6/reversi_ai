import torch

class TemperatureScheduler:
    def __init__(self, start: float, end: float, total_steps: int, cooling_phase_ratio: float = 1.0):
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.cooling_phase_ratio = cooling_phase_ratio
        self.current_temperature = start
        self.current_step = 0
        self.warinig_printed = False

    def get_temperature(self):
        return self.current_temperature

    def step(self):
        self.current_step += 1
        if self.current_step > self.total_steps and not self.warinig_printed:
            self.warinig_printed = True
            print("Warning: TemperatureScheduler step called after total_steps")
            return
        if self.current_step > self.total_steps * self.cooling_phase_ratio:
            return
        self.current_temperature = self.start + (self.end - self.start) * self.current_step / (self.total_steps * self.cooling_phase_ratio)
    
    def temp_teacher(self, teacher_v: torch.Tensor) -> torch.Tensor:
        return 0.5 + (teacher_v - 0.5) / self.current_temperature
