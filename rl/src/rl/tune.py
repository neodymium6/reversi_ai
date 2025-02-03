import optuna
from optuna.trial import Trial
from pprint import pprint
import json
from datetime import datetime
import os
import torch
from rl.agents import AgentConfig, Agent
from rl.agents.net_driver import NetType
from rl.agents.net_driver.cnn import CnnConfig
from rl.memory import MemoryType, MemoryConfig
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
BOARD_BATCH_SIZE = 240
EPISODES = 120000
EPS_START = 1.0
STEPS_PER_OPTIMIZE = 1
OPTIMIZE_PER_TARGET_UPDATE = 1

TUNE_DIR = "tune"
STUDY_NAME = "config_tuning_v2"
STORAGE_URL = f"sqlite:///{TUNE_DIR}/{STUDY_NAME}.db"
N_TRIALS = 60
RANDOM_SEED = 42

EVAL_N_GAMES = 1000
EVAL_WEIGHTS = {
    "random": 0.1,
    "alpha_beta": 0.7,
    "mcts": 0.2,
}

def get_config(trial: Trial) -> AgentConfig:
    trial_dir = Path(TUNE_DIR) / f"trial_{trial.number}"
    # 1 episode = approximately 60 moves (exept pass moves and early game end)
    # Total experiences during training will be approximately 60 * EPISODES
    # memory_ratio determines what fraction of total experiences we keep
    memory_ratio = trial.suggest_float("memory_ratio", 0.05, 10.0)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 1.0)
    memory_config = MemoryConfig(
        memory_size=int(EPISODES * memory_ratio),
        memory_type=MemoryType.PROPORTIONAL,
        alpha=alpha,
        beta=beta,
    )
    net_config = CnnConfig(
        num_channels=64,
        fc_hidden_size=256,
        net_type=NetType.Transformer,
    )
    n_board_init_random_moves = trial.suggest_int("n_board_init_random_moves", 4, 30)
    p_board_init_random_moves = trial.suggest_float("p_board_init_random_moves", 0.0, 1.0)
    eps_end = trial.suggest_float("eps_end", 0.01, 0.1)
    eps_decay = trial.suggest_int("eps_decay", 5, 30)
    lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    gradient_clip = trial.suggest_float("gradient_clip", 0.1, 5.0)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    config = AgentConfig(
        memory_config=memory_config,
        net_config=net_config,
        batch_size=BATCH_SIZE,
        board_batch_size=BOARD_BATCH_SIZE,
        n_board_init_random_moves=n_board_init_random_moves,
        p_board_init_random_moves=p_board_init_random_moves,
        device=DEVICE,
        eps_start=1.0,
        eps_end=eps_end,
        eps_decay=eps_decay,
        lr=lr,
        gradient_clip=gradient_clip,
        gamma=gamma,
        n_episodes=EPISODES,
        steps_per_optimize=STEPS_PER_OPTIMIZE,
        optimize_per_target_update=OPTIMIZE_PER_TARGET_UPDATE,
        verbose=False,
        model_path=str(trial_dir / "model.pth"),
    )
    return config

def save_trial_info(trial: Trial, filepath: str):
    """Save trial information to a JSON file"""
    trial_info = {
        "number": trial.number,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "datetime": datetime.now().isoformat()
    }
    
    with open(filepath, 'w') as f:
        json.dump(trial_info, f, indent=2)

def objective(trial: Trial) -> float:
    print(f"\nStarting Trial {trial.number}")
    trial_dir = Path(TUNE_DIR) / f"trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    config = get_config(trial)
    agent = Agent(config)
    save_trial_info(trial, trial_dir / "trial_info.json")
    try:
        # Train the agent
        print("Training agent...")
        agent.train()

        # Evaluate against different opponents
        print("Evaluating against random opponent...")
        random_win_rate = agent.vs_random(EVAL_N_GAMES)
        print(f"Random win rate: {random_win_rate:.3f}")
        
        print("Evaluating against alpha-beta opponent...")
        alpha_beta_win_rate = agent.vs_alpha_beta(EVAL_N_GAMES)
        print(f"Alpha-beta win rate: {alpha_beta_win_rate:.3f}")
        
        print("Evaluating against MCTS opponent...")
        mcts_win_rate = agent.vs_mcts(EVAL_N_GAMES)
        print(f"MCTS win rate: {mcts_win_rate:.3f}")
        
        # Store metrics
        trial.set_user_attr("random_win_rate", random_win_rate)
        trial.set_user_attr("alpha_beta_win_rate", alpha_beta_win_rate)
        trial.set_user_attr("mcts_win_rate", mcts_win_rate)
        
        # Combined score (weighted average)
        score = (
            random_win_rate * EVAL_WEIGHTS["random"] +
            alpha_beta_win_rate * EVAL_WEIGHTS["alpha_beta"] +
            mcts_win_rate * EVAL_WEIGHTS["mcts"]
        )

        # Save trial results
        trial_results = {
            "random_win_rate": random_win_rate,
            "alpha_beta_win_rate": alpha_beta_win_rate,
            "mcts_win_rate": mcts_win_rate,
            "score": score,
            "datetime": datetime.now().isoformat()
        }
        with open(trial_dir / "trial_results.json", "w") as f:
            json.dump(trial_results, f, indent=2)

        print(f"Trial {trial.number} score: {score:.3f}")

        # Clean up
        del agent
        torch.cuda.empty_cache()
        return score
    
    except Exception as e:
        # Clean up
        del agent
        torch.cuda.empty_cache()
        print(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.exceptions.TrialPruned()

def tune(resume: bool):
    if DEVICE == torch.device("cuda"):
        torch.set_float32_matmul_precision("high")
        print(f"Using CUDA, setting float32_matmul_precision to high")
    # Create directory for study results
    os.makedirs(TUNE_DIR, exist_ok=True)
    if resume:
        try:
            print("Resuming study...")
            study = optuna.load_study(
                study_name=STUDY_NAME,
                storage=STORAGE_URL,
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            )
            print(f"Loaded existing study with {len(study.trials)} trials")
        except Exception as e:
            print(f"Failed to resume study: {str(e)}")
            print("Changing mode to create new study...")
            resume = False

    if not resume:
        print("Creating new study...")
        try:
            study = optuna.create_study(
                study_name=STUDY_NAME,
                storage=STORAGE_URL,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
                load_if_exists=False,
            )
        except optuna.exceptions.DuplicatedStudyError:
            print("Study name already exists although the study was not resumed")
            print("If you want to resume the study, use the --resume flag")
            print("If you want to create a new study, change the study name or delete the existing study")
            raise optuna.exceptions.DuplicatedStudyError

    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    print("\nBest trial:")
    print("  Value:", study.best_trial.value)
    print("\nBest parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    print("\nBest trial metrics:")
    print(f"  Random win rate: {study.best_trial.user_attrs['random_win_rate']:.3f}")
    print(f"  Alpha-beta win rate: {study.best_trial.user_attrs['alpha_beta_win_rate']:.3f}")
    print(f"  MCTS win rate: {study.best_trial.user_attrs['mcts_win_rate']:.3f}")

    best_config = get_config(study.best_trial)
    print("\nBest config:")
    pprint(best_config)

    # Save study results
    study_results = {
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
        "best_metrics": study.best_trial.user_attrs,
        "datetime": datetime.now().isoformat()
    }
    with open(f"{TUNE_DIR}/{STUDY_NAME}_results.json", "w") as f:
        json.dump(study_results, f, indent=2)
