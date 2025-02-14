import numpy as np
import matplotlib.pyplot as plt


def circuit_system(t, state, params):
    i1, v1, i2, v2 = state
    R1, L1, C1, R2, L2, C2, M = params

    # input voltage (can be constant or time-dependent)
    V1 = 10 * np.sin(2 * np.pi * t)
    V2 = 5 * np.sin(2 * np.pi * t)

    # compute derivatives
    det = L1 * L2 - M * M
    di1dt = (L2 * (-R1 * i1 - v1 + V1) + M * (-R2 * i2 - v2 + V2)) / det
    dv1dt = i1 / C1
    di2dt = (L1 * (-R2 * i2 - v2 + V2) + M * (-R1 * i1 - v1 + V1)) / det
    dv2dt = i2 / C2

    return np.array([di1dt, dv1dt, di2dt, dv2dt])


def rk4_step(f, t, y, h, params):
    k1 = f(t, y, params)
    k2 = f(t + h / 2, y + h * k1 / 2, params)
    k3 = f(t + h / 2, y + h * k2 / 2, params)
    k4 = f(t + h, y + h * k3, params)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


R1, L1, C1 = 10.0, 0.1, 1e-6  # first circuit
R2, L2, C2 = 15.0, 0.15, 1.5e-6  # second circuit
M = 0.05  # mutual inductance
params = [R1, L1, C1, R2, L2, C2, M]

t_span = (0, 0.01)  # 10 milliseconds
h = 0.0001
t = np.arange(t_span[0], t_span[1], h)

y0 = np.array([0.0, 0.0, 0.0, 0.0])  # all initial values zero

solution = np.zeros((len(t), 4))
solution[0] = y0

for i in range(1, len(t)):
    solution[i] = rk4_step(circuit_system, t[i - 1], solution[i - 1], h, params)

# plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

ax1.plot(t * 1000, solution[:, 0], 'b-', label='Current 1')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Current (A)')
ax1.set_title('Current in Circuit 1')
ax1.grid(True)

ax2.plot(t * 1000, solution[:, 1], 'r-', label='Voltage 1')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Capacitor Voltage in Circuit 1')
ax2.grid(True)

ax3.plot(t * 1000, solution[:, 2], 'g-', label='Current 2')
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Current (A)')
ax3.set_title('Current in Circuit 2')
ax3.grid(True)

ax4.plot(t * 1000, solution[:, 3], 'm-', label='Voltage 2')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Voltage (V)')
ax4.set_title('Capacitor Voltage in Circuit 2')
ax4.grid(True)

plt.tight_layout()
plt.show()



