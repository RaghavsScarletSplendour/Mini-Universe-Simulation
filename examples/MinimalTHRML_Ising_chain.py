# https://chatgpt.com/c/6909e87c-4394-8333-bef6-620c242be815

import jax, jax.numpy as jnp
from thrml.pgm import SpinNode
from thrml.block_management import Block
from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init
from thrml.block_sampling import SamplingSchedule, sample_states

# 1) graph
N = 20
nodes = [SpinNode() for _ in range(N)]
edges = [(nodes[i], nodes[i+1]) for i in range(N-1)]

biases  = jnp.zeros((N,))
weights = jnp.ones((len(edges),)) * 0.5
beta    = jnp.array(1.0)
model   = IsingEBM(nodes, edges, biases, weights, beta)   # docs: IsingEBM

# 2) blocks: even/odd
free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, [])     # docs: IsingSamplingProgram

# 3) init with a *batch* of 1, then strip batch per block
key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_batched = hinton_init(k_init, model, free_blocks, (1,))   # returns list[ (1, block_size) ... ]
init_chain   = [arr[0] for arr in init_batched]                 # <- remove batch dim so each is (block_size,)

# 4) schedule and which nodes to *observe*
schedule = SamplingSchedule(n_warmup=100, n_samples=50, steps_per_sample=5)
nodes_to_sample = [Block(nodes)]                                # observe all nodes in one block

# 5) run
samples = sample_states(k_samp, program, schedule, init_chain, [], nodes_to_sample)

print(f"#samples: {len(samples)}")       # 1 list entry (because 1 Block in nodes_to_sample)
print(f"samples[0].shape: {samples[0].shape}")  # (n_samples, N)
