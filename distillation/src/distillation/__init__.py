import numpy as np
import h5py

DATA_PATH = "data/kifu.h5"

def load_data() -> np.ndarray:
    with h5py.File(DATA_PATH, "r") as f:
        data = f["data"][:]
        print(f"Loaded data with shape: {data.shape}")
        print(f"First 5 elements: {data[:5]}")


def main() -> None:
    print("Hello from distillation!")
    load_data()
