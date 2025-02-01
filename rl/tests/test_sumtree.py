from rl.memory.proportional import SumTree

def test_sumtree_initialization():
    # Test power of 2 capacity
    tree = SumTree(16)
    assert tree.capacity == 16
