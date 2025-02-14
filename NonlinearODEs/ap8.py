import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def van_der_pol_duffing(t, state, mu, alpha, beta, F, omega):
    x, y = state
    dx = y
    dy = -alpha*x - beta*x**3 + mu*(1 - x**2)*y + F*np.cos(omega*t)
    return [dx, dy]


mu = 1.0      # nonlinear damping parameter
alpha = -1.0  # linear stiffness
beta = 1.0    # nonlinear stiffness
F = 0.5       # forcing amplitude
omega = 1.2   # forcing frequency

# time span
t_span = (0, 100)
t_eval = np.linspace(*t_span, 1000)

# initial conditions
initial_state = [0.1, 0.1]

# tolerances for adaptive step size
rtol = 1e-6
atol = 1e-9

# solve using RK45 with adaptive step size
solution = solve_ivp(
    fun=lambda t, y: van_der_pol_duffing(t, y, mu, alpha, beta, F, omega),
    t_span=t_span,
    y0=initial_state,
    method='RK45',
    t_eval=t_eval,
    rtol=rtol,
    atol=atol
)

# plotting
plt.figure(figsize=(15, 5))

# time series
plt.subplot(121)
plt.plot(solution.t, solution.y[0], 'b-', label='Position (x)')
plt.plot(solution.t, solution.y[1], 'r-', label='Velocity (y)')
plt.xlabel('Time')
plt.ylabel('State Variables')
plt.title('Time Evolution')
plt.legend()
plt.grid(True)

# phase space
plt.subplot(122)
plt.plot(solution.y[0], solution.y[1], 'g-')
plt.xlabel('Position (x)')
plt.ylabel('Velocity (y)')
plt.title('Phase Space')
plt.grid(True)

# plot step sizes
plt.figure(figsize=(10, 4))
plt.plot(solution.t[:-1], np.diff(solution.t), 'k.')
plt.xlabel('Time')
plt.ylabel('Step Size')
plt.title('Adaptive Step Sizes')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Number of successful steps: {solution.nfev}")
print(f"Average step size: {np.mean(np.diff(solution.t)):.6f}")
print(f"Minimum step size: {np.min(np.diff(solution.t)):.6f}")
print(f"Maximum step size: {np.max(np.diff(solution.t)):.6f}")