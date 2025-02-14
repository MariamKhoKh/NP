import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**3 - 4*x**2 + x + 6


def f_prime_exact(x):
    return 3*x**2 - 8*x + 1


def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h


def backward_diff(f, x, h):
    return (f(x) - f(x - h)) / h


def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def higher_order_central_diff(f, x, h):
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


x0 = 2.0  # Example point to approximate the derivative
h_values = np.logspace(-9, -1, 50)  # Step sizes from 1e-9 to 1e-1

# Store errors for each method
errors_forward = []
errors_backward = []
errors_central = []
errors_higher_order = []

# Calculate errors for each h
for h in h_values:
    exact = f_prime_exact(x0)
    errors_forward.append(abs(forward_diff(f, x0, h) - exact))
    errors_backward.append(abs(backward_diff(f, x0, h) - exact))
    errors_central.append(abs(central_diff(f, x0, h) - exact))
    errors_higher_order.append(abs(higher_order_central_diff(f, x0, h) - exact))

# Plotting
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_forward, label="Forward Difference (1st Order)", marker='o')
plt.loglog(h_values, errors_backward, label="Backward Difference (1st Order)", marker='o', markersize=4)
plt.loglog(h_values, errors_central, label="Central Difference (2nd Order)", marker='o')
plt.loglog(h_values, errors_higher_order, label="Higher Order Central Difference (4th Order)", marker='o')
plt.xlabel("Step size h")
plt.ylabel("Error")
plt.title("Error vs Step Size for Finite Difference Methods (Polynomial Function)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()



