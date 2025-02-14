import numpy as np
import matplotlib.pyplot as plt


def food_chain_model(t, X, params):
    x, y, z = X
    alpha1, alpha2, beta1, beta2, d1, d2 = params

    dx = x * (1 - x) - (alpha1 * x * y) / (1 + beta1 * x)
    dy = (alpha1 * x * y) / (1 + beta1 * x) - (alpha2 * y * z) / (1 + beta2 * y) - d1 * y
    dz = (alpha2 * y * z) / (1 + beta2 * y) - d2 * z

    return np.array([dx, dy, dz])


def adams_bashforth_4(f, t, y0, params, h):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0

    # RK4 for first 3 steps
    for i in range(3):
        k1 = f(t[i], y[i], params)
        k2 = f(t[i] + h / 2, y[i] + h * k1 / 2, params)
        k3 = f(t[i] + h / 2, y[i] + h * k2 / 2, params)
        k4 = f(t[i] + h, y[i] + h * k3, params)
        y[i + 1] = y[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # AB4 method
    for i in range(3, n - 1):
        f_n = f(t[i], y[i], params)
        f_n1 = f(t[i - 1], y[i - 1], params)
        f_n2 = f(t[i - 2], y[i - 2], params)
        f_n3 = f(t[i - 3], y[i - 3], params)

        y[i + 1] = y[i] + h / 24 * (55 * f_n - 59 * f_n1 + 37 * f_n2 - 9 * f_n3)

    return y


alpha1, alpha2 = 5.0, 0.1
beta1, beta2 = 3.0, 2.0
d1, d2 = 0.4, 0.01
params = (alpha1, alpha2, beta1, beta2, d1, d2)

# time points
t0, tf = 0, 200
h = 0.01
t = np.arange(t0, tf, h)

# initial conditions
x0, y0, z0 = 0.5, 0.5, 0.5
initial_conditions = np.array([x0, y0, z0])

# solve using AB4
solution = adams_bashforth_4(food_chain_model, t, initial_conditions, params, h)


fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(121)
ax1.plot(t, solution[:, 0], 'g-', label='Resources')
ax1.plot(t, solution[:, 1], 'b-', label='Consumers')
ax1.plot(t, solution[:, 2], 'r-', label='Predators')
ax1.set_xlabel('Time')
ax1.set_ylabel('Population Density')
ax1.set_title('Population Dynamics')
ax1.legend()
ax1.grid(True)

# 3D phase space
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot3D(solution[:, 0], solution[:, 1], solution[:, 2], 'k-')
ax2.set_xlabel('Resources (x)')
ax2.set_ylabel('Consumers (y)')
ax2.set_zlabel('Predators (z)')
ax2.set_title('Phase Space')

plt.tight_layout()
plt.show()
