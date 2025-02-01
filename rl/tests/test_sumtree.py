from rl.memory.proportional import SumTree
import pytest
import numpy as np

def test_sumtree_initialization():
    # Test power of 2 capacity
    tree = SumTree(16)
    assert tree.capacity == 16

    with pytest.raises(AssertionError):
        SumTree(15)

def test_sumtree_add():
    tree = SumTree(16)
    tree.add(0.5)
    assert tree.length == 1
    assert tree.pointer == 1
    assert tree.max_p == 0.5
    idx = 15
    while idx >= 0:
        assert tree.tree[idx] == 0.5
        idx = (idx - 1) // 2

    tree.add(0.6)
    assert tree.length == 2
    assert tree.max_p == 0.6
    assert tree.pointer == 2
    assert tree.tree[15] == 0.5
    assert tree.tree[16] == 0.6
    idx = (16 - 1) // 2
    while idx >= 0:
        assert tree.tree[idx] == 1.1
        idx = (idx - 1) // 2

    assert tree.sum() == 1.1
    assert len(tree) == 2

def test_sumtree_update():
    tree = SumTree(16)
    tree.add(0.5)
    tree.add(0.6)
    tree.update(0, 0.7)
    assert tree.max_p == 0.7
    assert tree.tree[15] == 0.7
    idx = (16 - 1) // 2
    while idx >= 0:
        assert float(tree.tree[idx]) == 1.3
        idx = (idx - 1) // 2

def test_sumtree_cycling():
    tree = SumTree(4)
    # Fill the tree
    for i in range(4):
        tree.add(float(i + 1))
    assert tree.length == 4
    assert tree.pointer == 0
    
    # Add one more to test cycling
    tree.add(5.0)
    assert tree.length == 4
    assert tree.pointer == 1
    
    # Verify the old value was replaced
    leaf_start = tree.capacity - 1
    assert tree.tree[leaf_start] == 5.0  # New value
    assert tree.tree[leaf_start + 1] == 2.0  # Original second value

def test_sumtree_sampling():
    tree = SumTree(4)
    priorities = [0.1, 0.2, 0.3, 0.4]
    for p in priorities:
        tree.add(p)
    
    # Test basic sampling
    indices = tree.sample_indices(1000)
    counts = np.zeros(4)
    for idx in indices:
        counts[idx] += 1
    
    # Higher priorities should be sampled more frequently
    assert counts[3] > counts[0]  # Highest priority sampled more than lowest
    
    # All indices should be within valid range
    assert all(0 <= idx < tree.capacity for idx in indices)

def test_sumtree_max():
    tree = SumTree(4)
    # Test initial max (None case)
    assert tree.max() == 0.0
    
    # Test updating max through add
    tree.add(0.5)
    assert tree.max() == 0.5
    tree.add(0.7)
    assert tree.max() == 0.7
    tree.add(0.3)
    assert tree.max() == 0.7  # Max should not decrease
    
    # Test updating max through update
    tree.update(0, 0.9)
    assert tree.max() == 0.9
