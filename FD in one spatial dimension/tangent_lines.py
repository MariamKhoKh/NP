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
h = 0.1 # Step size for the finite difference approximation()

# Compute exact and approximate derivatives at x0
exact_slope = f_prime_exact(x0)
approx_slope_forward = forward_diff(f, x0, h)
approx_slope_backward = backward_diff(f, x0, h)
approx_slope_central = central_diff(f, x0, h)
approx_slope_higher_order = higher_order_central_diff(f, x0, h)


# Define tangent line functions for exact and approximate slopes
def tangent_line_exact(x):
    return f(x0) + exact_slope * (x - x0)


def tangent_line_forward(x):
    return f(x0) + approx_slope_forward * (x - x0)


def tangent_line_backward(x):
    return f(x0) + approx_slope_backward * (x - x0)


def tangent_line_central(x):
    return f(x0) + approx_slope_central * (x - x0)


def tangent_line_higher_order(x):
    return f(x0) + approx_slope_higher_order * (x - x0)


# Plot the function and tangent lines
x_vals = np.linspace(x0 - 0.5, x0 + 0.5, 100)  # Range of x values around x0
plt.figure(figsize=(12, 8))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 4x^2 + x + 6", color="blue")
plt.plot(x_vals, tangent_line_exact(x_vals), '--', label="Exact Tangent Line", color="green", linewidth=2)
plt.plot(x_vals, tangent_line_forward(x_vals), '--', label="Forward Diff Tangent Line", color="red")
plt.plot(x_vals, tangent_line_backward(x_vals), '--', label="Backward Diff Tangent Line", color="orange")
plt.plot(x_vals, tangent_line_central(x_vals), '--', label="Central Diff Tangent Line", color="violet", linewidth=3)
plt.plot(x_vals, tangent_line_higher_order(x_vals), '--', label="Higher Order Central Diff Tangent Line", color="pink")

# Mark the point of tangency at x0
plt.scatter(x0, f(x0), color="black", zorder=5, label="Point of Tangency at x = {:.2f}".format(x0))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.title("Function and Tangent Line Comparison at x = {:.2f}".format(x0))
plt.grid(True)
plt.show()
