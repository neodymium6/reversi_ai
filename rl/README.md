# Reversi AI using Deep Reinforcement Learning

A Reversi AI implementation that combines deep reinforcement learning techniques with a high-performance Rust-based game engine. The project leverages PyTorch for neural network training and implements various deep learning architectures.

## Features

### Core AI Components
- Deep Q-Learning implementation with multiple neural network architectures:
  - Convolutional Neural Networks (CNN) for board state processing
  - ResNet architecture for deeper feature extraction
  - Fully Connected Network option for comparison
- Dueling network architecture that separates state value and action advantage estimation
- Experience replay memory system with both uniform and prioritized sampling
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

- Python 3.13 or higher (required)
- CUDA-capable GPU (optional but recommended)
- Core dependencies: See `pyproject.toml` for full list

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:neodymium6/reversi_ai.git
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
│   │   └── net_driver/      # Neural network driver implementations
│   │       ├── cnn.py       # CNN network driver
│   │       ├── dense.py     # Dense network driver
│   │       └── q_net.py     # Base Q-learning driver
│   ├── memory/          # Experience replay system
│   │   ├── proportional.py  # Prioritized experience replay
│   │   └── simple.py       # Basic replay buffer
│   ├── models/          # Neural network architectures
│   │   ├── cnn.py          # Basic CNN model
│   │   ├── cnn_dueling.py  # Dueling CNN architecture
│   │   ├── dense.py        # Fully connected network
│   │   └── resnet.py       # ResNet implementation
│   └── __init__.py      # Configuration and entry points
├── tests/              # Test suite
│   ├── __init__.py
│   └── test_sumtree.py  # Tests for sum tree data structure
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

Default parameters (from `src/rl/__init__.py`):

```python
# Training parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 120000
BATCH_SIZE = 512

# Memory configuration
memory_config = MemoryConfig(
    memory_size=EPISODES // 5,  # Automatically scaled with total episodes
    memory_type=MemoryType.PROPORTIONAL,
    alpha=0.5,
    beta=0.5,
)

# Model parameters
net_config = CnnConfig(
    num_channels=64,
    fc_hidden_size=256,
    net_type=NetType.RESNET10,
)

train_config = AgentConfig(
    memory_config=memory_config,
    net_config=net_config,
    batch_size=BATCH_SIZE,
    board_batch_size=240,
    n_board_init_random_moves=10,
    p_board_init_random_moves=0.5,
    device=DEVICE,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=10,
    lr=1e-5,
    gradient_clip=1.0,
    gamma=0.99,
    n_episodes=EPISODES,
    steps_per_optimize=1,
    optimize_per_target_update=1,
    verbose=True,
    model_path="cnn_agent.pth",
)
```

#### Note

* Training time will vary significantly depending on your hardware configuration.
* The win rate against the random player is around 95% after successful training.
* In my environment (RTX 4070Ti), training the ResNet model for 120,000 episodes took around 20 minutes.

### Experience Replay
- Supports both uniform and prioritized sampling strategies
- SumTree implementation for efficient prioritized sampling
- Configurable memory size and priority parameters
- Supports efficient batch sampling
- Dynamic priority updates based on TD error

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
   - Sample batches from replay memory (uniform or prioritized)
   - Compute Q-learning targets using target network
   - Update policy network via gradient descent
   - Periodically update target network
   - Update priorities for sampled transitions

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
