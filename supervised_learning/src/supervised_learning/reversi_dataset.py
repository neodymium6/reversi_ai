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
from supervised_learning.models import ReversiNet
from multiprocessing import shared_memory
from typing import Type

def board_to_input(board: Board, model_class: Type[ReversiNet]) -> np.ndarray:
    board_tensor: torch.Tensor = model_class.board_to_input(board)
    return board_tensor.numpy()

def preprocess_chunk(
    chunk_start: int,
    chunk_end: int,
    shared_memory_name: str,
    shm_shape: Tuple[int, ...],
    shm_dtype: np.dtype,
    temp_dir: str,
    model_class: Type[ReversiNet],
) -> Tuple[int, int, str]:
    existing_shm = shared_memory.SharedMemory(name=shared_memory_name)
    shared_data = np.ndarray(shm_shape, dtype=shm_dtype, buffer=existing_shm.buf)
    board_to_input_func = functools.partial(board_to_input, model_class=model_class)

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
        return chunk_start, chunk_end, temp_path
    finally:
        del temp_memmap
        existing_shm.close()

class ReversiDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            model_class: Type[ReversiNet],
            chunk_size: int = int(1e6),
            shuffle: bool = True,
            preprocess_workers: int = 1,
            verbose: bool = True,
        ):
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.temp_dir = "tmp"
        self.model_class = model_class
        self.preprocess_workers = preprocess_workers
        self.verbose = verbose
        os.makedirs(self.temp_dir, exist_ok=True)

        self.scores = torch.tensor([], dtype=torch.float32)
        sample_board = Board()
        sample_tensor = torch.from_numpy(board_to_input(sample_board, self.model_class))
        self.tensor_shape = sample_tensor.shape
        self.mmap_path = os.path.join(self.temp_dir, f"tensor_cache_{uuid.uuid4()}.dat")
        self.mmap_tensors = np.memmap(
            self.mmap_path,
            dtype=np.float32,
            mode="w+",
            shape=(0, *self.tensor_shape),
        )

    def append_data(self, X: List[Tuple[int, int, int]]):
        last_idx = len(self)
        total_shape = (len(X) + len(self), *self.tensor_shape)
        total_size = np.prod(total_shape)
        if self.verbose:
            print(f"Appending {len(X)} samples to memmap, total shape: {total_shape}, total size: {total_size * 4 / (1024**3):.2f} GB")
        self.scores = torch.cat([self.scores, torch.tensor([x[2] for x in X], dtype=torch.float32)])
        appending_byte_size = len(X) * np.prod(self.tensor_shape) * 4
        del self.mmap_tensors
        os.truncate(self.mmap_path, os.path.getsize(self.mmap_path) + appending_byte_size)
        self.mmap_tensors = np.memmap(
            self.mmap_path,
            dtype=np.float32,
            mode='r+',
            shape=total_shape
        )
        preprocess_minibatch_size = self.get_preprocess_minibatch_size(len(X))
        chunk_indices = [(i, min(i + preprocess_minibatch_size, len(X))) for i in range(0, len(X), preprocess_minibatch_size)]
        po = [(x[0], x[1]) for x in X]
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
            model_class=self.model_class,
        )
        del po_np
        with ProcessPoolExecutor(max_workers=self.preprocess_workers) as executor:
            futures = [
                executor.submit(preprocess_func, chunk_start, chunk_end)
                for chunk_start, chunk_end in chunk_indices
            ]
            for future in tqdm.tqdm(futures, desc="Collecting results", leave=False):
                chunk_start, chunk_end, temp_path = future.result()
                temp_memmap = np.memmap(
                    temp_path,
                    dtype=np.float32,
                    mode="r",
                    shape=(chunk_end - chunk_start, *self.tensor_shape)
                )
                self.mmap_tensors[last_idx + chunk_start:last_idx + chunk_end] = temp_memmap
                self.mmap_tensors.flush()
                del temp_memmap
                os.remove(temp_path)
        if self.verbose:
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

    def get_preprocess_minibatch_size(self, data_len: int) -> int:
        preprocess_minibatch_size = data_len // self.preprocess_workers
        preprocess_minibatch_size = max(preprocess_minibatch_size, int(1e4))
        preprocess_minibatch_size = min(preprocess_minibatch_size, int(1e6))
        return preprocess_minibatch_size

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

        for chunk_start in range(0, len(indices), self.chunk_size):
            chunk_indices = indices[chunk_start:chunk_start + self.chunk_size]
            memory_tensor = torch.from_numpy(self.mmap_tensors[chunk_indices].copy())
            scores = self.scores[chunk_indices]
            yielding_indices = list(range(len(chunk_indices)))
            if self.shuffle:
                random.shuffle(yielding_indices)
            for idx in yielding_indices:
                tensor = memory_tensor[idx]
                score = scores[idx]
                yield tensor, score
            del memory_tensor
            del chunk_indices
            del scores
            del yielding_indices

    def __del__(self):
        if hasattr(self, 'mmap_tensors'):
            del self.mmap_tensors
        if hasattr(self, 'mmap_path') and os.path.exists(self.mmap_path):
            try:
                os.remove(self.mmap_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {self.mmap_path}: {e}")
