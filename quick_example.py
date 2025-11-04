# universe_anim.py
# --- pick a GUI backend BEFORE pyplot import (critical) ---
import sys, matplotlib
if sys.platform == "darwin":
    try:
        matplotlib.use("macosx")   # works with python.org mac build
    except Exception:
        matplotlib.use("TkAgg")    # fallback (needs Tk installed)
else:
    matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt

N, T = 60, 1.8
state = np.random.choice([-1, 1], (N, N))

def dE(i, j):
    s = state[i, j]
    nb = state[(i+1)%N, j] + state[(i-1)%N, j] + state[i, (j+1)%N] + state[i, (j-1)%N]
    return 2 * s * nb

fig, ax = plt.subplots()
im = ax.imshow(state, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks([]); ax.set_yticks([])

def step(k=2000):
    for _ in range(k):
        i, j = np.random.randint(0, N, 2)
        de = dE(i, j)
        if de < 0 or np.random.rand() < np.exp(-de / T):
            state[i, j] *= -1

def on_tick(*_):
    step(800)                 # do work between frames
    im.set_data(state)
    fig.canvas.draw_idle()

timer = fig.canvas.new_timer(interval=30)
timer.add_callback(on_tick)
timer.start()

plt.show()  # <-- blocks and keeps the window open
