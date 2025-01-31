# Reversi AI using Deep Reinforcement Learning

A Reversi AI implementation that combines deep reinforcement learning techniques with a high-performance Rust-based game engine. The project leverages PyTorch for neural network training and implements various deep learning architectures.

## Features

### Core AI Components
- Deep Q-Learning implementation with multiple neural network architectures:
  - Convolutional Neural Networks (CNN) for board state processing
  - ResNet architecture for deeper feature extraction
  - Fully Connected Network option for comparison
- Dueling network architecture that separates state value and action advantage estimation
- Experience replay memory system for stable learning
- Efficient batch processing of multiple games simultaneously

### Training & Evaluation
- Self-play training mechanism
- Performance evaluation against multiple opponents:
  - Random move player for baseline testing
  - Alpha-beta pruning based opponent for advanced evaluation
- Progressive learning with epsilon-greedy exploration
- Support for model checkpointing and training visualization

### Technical Implementation
- Rust-based Reversi engine for fast game simulation
- PyTorch integration for deep learning
- CUDA support for GPU acceleration
- Modular architecture for easy experimentation with different network designs

## Requirements

- Python 3.13 or higher
- CUDA-capable GPU (optional but recommended)
- Core dependencies:　See `pyproject.toml` for full list

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:neodymium6/rust_reversi.git
   cd reversi_ai/rl
   ```

2. Install the package:
   ```bash
   uv sync
   ```

## Usage

The project provides three main command-line interfaces:

```bash
# Start training a new model
uv run train

# Evaluate model against random opponent
uv run vs_random

# Evaluate model against alpha-beta pruning opponent
uv run vs_alpha_beta
```

## Project Structure

```
rl/
├── src/rl/
│   ├── agents/          # AI agent implementations
│   │   ├── batch_board.py   # Efficient batch game processing
│   │   ├── cnn.py          # CNN-based agent
│   │   ├── dense.py        # Dense network agent
│   │   └── q_net.py        # Base Q-learning implementation
│   ├── memory/          # Experience replay system
│   │   └── simple.py       # Basic replay buffer
│   ├── models/          # Neural network architectures
│   │   ├── cnn.py          # Basic CNN model
│   │   ├── cnn_dueling.py  # Dueling CNN architecture
│   │   ├── dense.py        # Fully connected network
│   │   └── resnet.py       # ResNet implementation
│   └── __init__.py      # Configuration and entry points
```

## Technical Details

### Neural Network Architectures

#### ResNet Model
- 10 residual blocks with skip connections
- Dual output streams for dueling architecture:
  - Value stream for state evaluation
  - Advantage stream for action evaluation
- Batch normalization for stable training
- ReLU activation functions

#### Training Configuration
Default parameters (configurable in `src/rl/__init__.py`):
```python
memory_size = 100000
batch_size = 512
board_batch_size = 240
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 10
learning_rate = 1e-5
gamma = 0.99
n_episodes = 480000
episodes_per_optimize=16,
episodes_per_target_update=128,
num_channels=64,
fc_hidden_size=256,
```

##### Note

* The training ResNet10 with this configuration takes approximately 30 minutes on core i7 13700 and RTX 4070Ti.
* The win rate against the random player is around 95% after 480,000 episodes.


### Experience Replay
- Dequeue buffer implementation
- Uniform random sampling
- Configurable memory size
- Supports efficient batch sampling

### Training Process

1. **Initialization**
   - Initialize neural networks (policy and target)
   - Setup experience replay memory
   - Configure training parameters

2. **Game Generation**
   - Batch process multiple games simultaneously
   - Use epsilon-greedy strategy for exploration
   - Store transitions in replay memory

3. **Learning**
   - Sample batches from replay memory
   - Compute Q-learning targets using target network
   - Update policy network via gradient descent
   - Periodically update target network

4. **Evaluation**
   - Regular evaluation against random and alpha-beta opponents
   - Track win rates and learning curves
   - Save model checkpoints

## Performance Monitoring

The training process includes:
- Real-time loss tracking
- Win rate monitoring against random player
- Automatic model checkpointing
- Learning curve visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License
