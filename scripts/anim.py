import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load tensors
x_test = torch.load("./results/x_test.pt")
test_x_fm = torch.load("./results/test_x_fm.pt")
source = torch.load("./results/source.pt")

# Parameters
Nsamples, dim = x_test.shape
_, nsteps, _ = test_x_fm.shape
nrows, ncols = 3, 3
K = nrows * ncols

# Random sample indices
indices = np.random.choice(Nsamples, K, replace=False)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
axes = axes.flatten()
for i, ax in zip(indices, axes):
    plt.sca(ax)
    plt.plot(source[i, :])
plt.savefig("x_start.pdf")
plt.close

# Setup plot
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
axes = axes.flatten()

lines = []
for i, ax in zip(indices, axes):
    ax.set_title(f"Sample {i}")
    ax.set_xlim(0, dim)
    signal_min = min(x_test[i].min().item(), test_x_fm[i].min().item())
    signal_max = max(x_test[i].max().item(), test_x_fm[i].max().item())
    ax.set_ylim(signal_min, signal_max)

    (ref_line,) = ax.plot(range(dim), x_test[i].numpy(), "r-", label=r"$x$")
    (fm_line,) = ax.plot([], [], "b-", label=r"$x|y$")
    ax.legend()
    lines.append(fm_line)


# Animation update function
def update(frame):
    for line, i in zip(lines, indices):
        line.set_data(range(dim), test_x_fm[i, frame].numpy())
    return lines


ani = animation.FuncAnimation(fig, update, frames=nsteps, blit=True, interval=200, repeat=True)

plt.tight_layout()
# Save the animation as an mp4 file
ani.save("signal_animation.mp4", writer="ffmpeg", fps=5)
