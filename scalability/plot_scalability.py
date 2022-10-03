#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

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
ax1.plot(samples, problem_setup_time, label="Problem")
ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel("Samples")
ax2.set_ylabel("Solve time (ms)")
ax2.grid(visible=True)
ax2.plot(samples, casadi_solve_time, label="CasADi")
ax2.plot(samples, problem_solve_time, label="Problem")
ax2.legend()

plt.show()
