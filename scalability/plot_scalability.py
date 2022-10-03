#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_line_fit(ax, x, y, color):
    def line(x, m, b):
        return m * x + b

    # Fit line y = mx + b to x-y data
    m, b = curve_fit(line, x, y, p0=(1, 1))[0]

    label = f"Fit: y = {m:.4g}x"
    if b > 0:
        label += f" + {b:.4g}"
    else:
        label += f" - {abs(b):.4g}"

    ax.plot(x, line(x, m, b), color=color, label=label, linestyle="--")


def plot_exp_fit(ax, x, y, color):
    def exp(x, a, b, c):
        return a * np.exp(b * x) + c

    # Fit exponential y = aeᵇˣ + c to x-y data
    a, b, c = curve_fit(exp, x, y, p0=(1, 1e-6, 1))[0]

    label = f"Fit: y = {a:.4g}e^({b:.4g}x)"
    if c > 0:
        label += f" + {c:.4g}"
    else:
        label += f" - {abs(c):.4g}"

    ax.plot(x, exp(x, a, b, c), color=color, label=label, linestyle="--")


data = np.genfromtxt("results.csv", delimiter=",", skip_header=1)
samples = data[:, 0].T
casadi_setup_time = data[:, 1].T
casadi_solve_time = data[:, 2].T
problem_setup_time = data[:, 3].T
problem_solve_time = data[:, 4].T

fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("Optimization API runtime vs samples")
ax1.set_ylabel("Setup time (ms)")
ax1.grid(visible=True)

ax1.plot(samples, casadi_setup_time, label="CasADi")
plot_line_fit(ax1, samples, casadi_setup_time, color="blue")

ax1.plot(samples, problem_setup_time, label="Problem")
plot_exp_fit(ax1, samples, problem_setup_time, color="orange")

ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel("Samples")
ax2.set_ylabel("Solve time (ms)")
ax2.grid(visible=True)

ax2.plot(samples, casadi_solve_time, label="CasADi")
plot_exp_fit(ax2, samples, casadi_solve_time, color="blue")

ax2.plot(samples, problem_solve_time, label="Problem")
plot_exp_fit(ax2, samples, problem_solve_time, color="orange")

ax2.legend()

plt.show()
