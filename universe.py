import numpy as np
import matplotlib.pyplot as plt
import thrml as th

# Grid size
N = 30

# Initialize spins randomly
state = np.random.choice([-1, 1], size=(N, N))

# Energy function: neighbours want to align
def local_energy(i, j, state, J=1.0):
    s = state[i, j]
    # Periodic boundary conditions
    neighbors = [state[(i+1)%N,j], state[(i-1)%N,j], state[i,(j+1)%N], state[i,(j-1)%N]]
    return -J * s * sum(neighbors)

# Applies one Metropolis update — 
#   randomly picks a spin, computes how much the system’s energy would change if you flipped it, 
#   and decides whether to flip based on temperature.
# This implements the Metropolis algorithm, a Monte Carlo method used to simulate how systems evolve toward equilibrium.

def metropolis_step(state, T=2.5):
    i, j = np.random.randint(0, N, 2)
    dE = -2 * local_energy(i, j, state)
    if dE < 0 or np.random.rand() < np.exp(-dE/T):
        state[i, j] *= -1
    return state

# Simulation loop
plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(state, cmap="coolwarm", interpolation="nearest")

for step in range(1000):
    state = metropolis_step(state, T=2.0)
    if step % 10 == 0:
        im.set_data(state)
        plt.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()
