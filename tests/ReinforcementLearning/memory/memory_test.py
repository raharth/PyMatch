import torch
from pymatch.ReinforcementLearning.memory import Memory

memory = Memory(
    memory_cell_names=["test1", "test2"],
    n_samples=5,
    memory_size=10,
    batch_size=2,
)

for i in range(10):
    memory.memorize([torch.tensor([[10*i + 1]]), torch.tensor([[10*i + 2]])], ["test1", "test2"])

print(memory.memory)

memory[3]
