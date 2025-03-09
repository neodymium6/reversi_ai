import os
import uuid
import random
import math
import functools
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
import tqdm
from typing import List, Tuple
from rust_reversi import Board, Turn
from supervised_learning.models.dense import DenseNet
from multiprocessing import shared_memory

def board_to_input(board: Board) -> np.ndarray:
    board_tensor = DenseNet.board_to_input(board)
    return board_tensor.numpy()

def preprocess_chunk(
    chunk_start: int,
    chunk_end: int,
    shared_memory_name: str,
    shm_shape: Tuple[int, ...],
    shm_dtype: np.dtype,
    temp_dir: str,
    board_to_input_func,
) -> Tuple[int, int, np.ndarray]:
    existing_shm = shared_memory.SharedMemory(name=shared_memory_name)
    shared_data = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)

    temp_path = os.path.join(temp_dir, f"temp_chunk_{chunk_start}_{chunk_end}_{uuid.uuid4()}.dat")
    chunk_size = chunk_end - chunk_start
    board = Board()
    board.set_board(shared_data[0][0], shared_data[0][1], Turn.BLACK)
    sample_shape = board_to_input_func(board).shape
    temp_memmap = np.memmap(
        temp_path,
        dtype=np.float32,
        mode="w+",
        shape=(chunk_size, *sample_shape)
    )
    try:
        for i, idx in enumerate(range(chunk_start, chunk_end)):
            board = Board()
            board.set_board(shared_data[idx][0], shared_data[idx][1], Turn.BLACK)
            temp_memmap[i] = board_to_input_func(board)
        temp_memmap.flush()
        result = temp_memmap.copy()
        return chunk_start, chunk_end, result
    finally:
        del temp_memmap
        existing_shm.close()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {e}")

class ReversiDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            X: List[Tuple[Board, int]],
            chunk_size: int = int(1e5),
            shuffle: bool = True,
            preprocess_workers: int = 1
        ):
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.temp_dir = "tmp"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.scores = torch.tensor([x[1] for x in X], dtype=torch.float32)

        sample_tensor = torch.from_numpy(board_to_input(X[0][0]))
        self.tensor_shape = sample_tensor.shape
        self.mmap_path = os.path.join(self.temp_dir, f"tensor_cache_{uuid.uuid4()}.dat")

        total_shape = (len(X), *self.tensor_shape)
        total_size = np.prod(total_shape)
        print(f"Creating memmap of shape {total_shape} ({total_size * 4 / (1024**3):.2f} GB)")

        self.mmap_tensors = np.memmap(
            self.mmap_path,
            dtype=np.float32,
            mode="w+",
            shape=total_shape,
        )

        print(f"Computing and storing board representations using {preprocess_workers} workers")
        preprocess_minibatch_size = len(X) // preprocess_workers
        preprocess_minibatch_size = max(preprocess_minibatch_size, int(1e4))
        preprocess_minibatch_size = min(preprocess_minibatch_size, int(5e6))
        chunk_indices = [(i, min(i + preprocess_minibatch_size, len(X))) for i in range(0, len(X), preprocess_minibatch_size)]
        # convert to (player_board, opponent_board) for pickle-able
        po = [(b.get_board()[0], b.get_board()[1]) for b, _s in X]
        del X
        po_np = np.array(po, dtype=np.uint64)
        shm = shared_memory.SharedMemory(create=True, size=po_np.nbytes)
        po_np_shared = np.ndarray(po_np.shape, dtype=po_np.dtype, buffer=shm.buf)
        po_np_shared[:] = po_np
        preprocess_func = functools.partial(
            preprocess_chunk,
            shared_memory_name=shm.name,
            shm_shape=po_np.shape,
            shm_dtype=po_np.dtype,
            temp_dir=self.temp_dir,
            board_to_input_func=board_to_input,
        )
        with ProcessPoolExecutor(max_workers=preprocess_workers) as executor:
            futures = [
                executor.submit(preprocess_func, chunk_start, chunk_end)
                for chunk_start, chunk_end in chunk_indices
            ]
            for future in tqdm.tqdm(futures, desc="Collecting results", leave=False):
                chunk_start, chunk_end, data = future.result()
                self.mmap_tensors[chunk_start:chunk_end] = data
                self.mmap_tensors.flush()
        print(f"Completed preprocessing. Data stored at {self.mmap_path}")

        del self.mmap_tensors
        self.mmap_tensors = np.memmap(
            self.mmap_path,
            dtype=np.float32,
            mode='r',
            shape=total_shape
        )
        shm.close()
        shm.unlink()
    
    def __len__(self) -> int:
        return len(self.scores)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx = 0
            end_idx = len(self)
        else:
            per_worker = int(math.ceil(len(self) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self))
        indices = list(range(start_idx, end_idx))
        if self.shuffle:
            random.shuffle(indices)

        for chunk_start in range(0, len(indices), self.chunk_size):
            chunk_indices = indices[chunk_start:chunk_start + self.chunk_size]
            memory_tensor = torch.from_numpy(self.mmap_tensors[chunk_indices].copy())
            for mem_idx, idx in enumerate(chunk_indices):
                tensor = memory_tensor[mem_idx]
                score = self.scores[idx]
                yield tensor, score

    def __del__(self):
        if hasattr(self, 'mmap_tensors'):
            del self.mmap_tensors
        if hasattr(self, 'mmap_path') and os.path.exists(self.mmap_path):
            try:
                os.remove(self.mmap_path)
                print(f"Cleaned up temporary file: {self.mmap_path}")
            except Exception as e:
                print(f"Warning: Could not delete temporary file {self.mmap_path}: {e}")
