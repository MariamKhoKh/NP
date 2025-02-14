import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


p1 = 0.01  # Insulin-independent glucose utilization
p2 = 0.015  # Insulin-dependent glucose utilization
p3 = 1.0  # Glucose production rate
p4 = 0.1  # Insulin degradation
p5 = 0.005  # Insulin production rate

# DIRK method coefficients
a21 = 0.2141
a31 = 0.2141
a32 = 0.5000
a33 = 0.2859
c1 = 0.4359
c2 = 0.7141
c3 = 1.0000


def f(t, y):
    """Right-hand side of the ODE system"""
    G, I = y
    return np.array([
        -p1 * G - p2 * G * I + p3,
        -p4 * I + p5 * G * G
    ])


def newton_stage(y_n, h, t_n, c, a_diag, k_prev=None, a_prev=None):
    """Newton iteration for a single stage"""

    def stage_equation(k):
        if k_prev is None:
            y = y_n + h * a_diag * k
        else:
            y = y_n.copy()
            for k_i, a_i in zip(k_prev, a_prev):
                y += h * a_i * k_i
            y += h * a_diag * k
        return k - f(t_n + c * h, y)

    k_guess = np.zeros(2)
    k = fsolve(stage_equation, k_guess)
    return k


def dirk_step(y_n, h, t_n):
    """Single step of the 3-stage DIRK method"""
    # stage 1
    k1 = newton_stage(y_n, h, t_n, c1, 0.4359)

    # stage 2
    k2 = newton_stage(y_n, h, t_n, c2, 0.5, [k1], [a21])

    # stage 3
    k3 = newton_stage(y_n, h, t_n, c3, a33, [k1, k2], [a31, a32])

    # compute solution
    y_next = y_n + h * (0.2141 * k1 + 0.5 * k2 + 0.2859 * k3)

    return y_next


# simulation parameters
t0 = 0
tf = 100
h = 0.1
t = np.arange(t0, tf + h, h)
N = len(t)

# initial conditions
G0 = 8.0
I0 = 0.5
y = np.zeros((N, 2))
y[0] = [G0, I0]

# time integration
for i in range(N - 1):
    y[i + 1] = dirk_step(y[i], h, t[i])


plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(t, y[:, 0], 'b-', label='Glucose')
plt.xlabel('Time')
plt.ylabel('Glucose Concentration')
plt.legend()
plt.grid(True)
plt.title('Glucose vs Time')

plt.subplot(132)
plt.plot(t, y[:, 1], 'r-', label='Insulin')
plt.xlabel('Time')
plt.ylabel('Insulin Concentration')
plt.legend()
plt.grid(True)
plt.title('Insulin vs Time')

plt.subplot(133)
plt.plot(y[:, 0], y[:, 1], 'g-')
plt.xlabel('Glucose Concentration')
plt.ylabel('Insulin Concentration')
plt.grid(True)
plt.title('Phase Portrait')

plt.tight_layout()
plt.show()


# Stability Analysis
def equilibrium_residual(y):
    return f(0, y)


eq_point = fsolve(equilibrium_residual, [G0, I0])
print(f"Equilibrium point: Glucose = {eq_point[0]:.3f}, Insulin = {eq_point[1]:.3f}")
