
TRANSFORMER_CONFIGS = {
    "balanced": {
        "patch_size": 2,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    },
    "deep_narrow": {
        "patch_size": 2,
        "embed_dim": 128,
        "num_heads": 4,
        "num_layers": 16,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    },
    "shallow_wide": {
        "patch_size": 2,
        "embed_dim": 384,
        "num_heads": 12,
        "num_layers": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    },
    "large_patch": {
        "patch_size": 4,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 8,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    },
    "mlp_heavy": {
        "patch_size": 2,
        "embed_dim": 160,
        "num_heads": 5,
        "num_layers": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
    },
    "attention_heavy": {
        "patch_size": 2,
        "embed_dim": 192,
        "num_heads": 16,
        "num_layers": 8,
        "mlp_ratio": 1.0,
        "dropout": 0.0,
    },
    "local_focus": {
        "patch_size": 2,
        "embed_dim": 256,
        "num_heads": 32,
        "num_layers": 6,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    },
    "global_focus": {
        "patch_size": 4,
        "embed_dim": 256,
        "num_heads": 4,
        "num_layers": 8,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
    },
}

DEFAULT_TRANSFORMER_CONFIG = TRANSFORMER_CONFIGS["mlp_heavy"]
