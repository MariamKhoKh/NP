import matplotlib
import numpy as np
import time

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
matplotlib.use('TkAgg')



class NumericalAnalysis:
    def __init__(self, g=-9.81):
        self.g = g

    def equations_of_motion(self, t, state, k_m_ratio):
        """
        equations of motion with gravity and air resistance
        state = [x, y, vx, vy]
        """
        _, _, vx, vy = state
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)

        drag_x = -k_m_ratio * v_magnitude * vx
        drag_y = -k_m_ratio * v_magnitude * vy

        return np.array([
            vx,
            vy,
            drag_x,  # x acceleration from air resistance
            self.g + drag_y  # y acceleration from gravity and air resistance
        ])

    def euler_step(self, t, state, dt, k_m_ratio):
        """Single step of Euler method"""
        derivatives = self.equations_of_motion(t, state, k_m_ratio)
        return state + dt * derivatives

    def rk4_step(self, t, state, dt, k_m_ratio):
        """Single step of RK4 method"""
        k1 = self.equations_of_motion(t, state, k_m_ratio)
        k2 = self.equations_of_motion(t + 0.5 * dt, state + 0.5 * dt * k1, k_m_ratio)
        k3 = self.equations_of_motion(t + 0.5 * dt, state + 0.5 * dt * k2, k_m_ratio)
        k4 = self.equations_of_motion(t + dt, state + dt * k3, k_m_ratio)

        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve_numerical(self, method, t_span, initial_state, dt, k_m_ratio):
        """Solve using either Euler or RK4 method"""
        t_current = t_span[0]
        t_end = t_span[1]
        state = initial_state.copy()

        t_points = []
        states = []

        start_time = time.perf_counter()

        while t_current <= t_end:
            t_points.append(t_current)
            states.append(state.copy())

            if method == 'euler':
                state = self.euler_step(t_current, state, dt, k_m_ratio)
            elif method == 'rk4':
                state = self.rk4_step(t_current, state, dt, k_m_ratio)

            t_current += dt

        runtime = time.perf_counter() - start_time

        return np.array(t_points), np.array(states), runtime

    def get_reference_solution(self, t_span, initial_state, k_m_ratio, t_eval):
        """Get high-accuracy reference solution using scipy's solve_ivp"""
        sol = solve_ivp(
            fun=lambda t, y: self.equations_of_motion(t, y, k_m_ratio),
            t_span=t_span,
            y0=initial_state,
            method='RK45',
            rtol=1e-12,
            atol=1e-12,
            t_eval=t_eval
        )
        return sol.t, sol.y.T

    def calculate_errors(self, numerical_solution, reference_solution):
        """Calculate various error metrics"""
        absolute_error = np.abs(numerical_solution - reference_solution)
        relative_error = absolute_error / (np.abs(reference_solution) + 1e-10)

        max_absolute_error = np.max(absolute_error)
        max_relative_error = np.max(relative_error)
        rmse = np.sqrt(np.mean(absolute_error ** 2))

        return {
            'max_absolute_error': max_absolute_error,
            'max_relative_error': max_relative_error,
            'rmse': rmse
        }


def run_analysis(t_span=(0, 10), initial_state=np.array([0, 0, 10, 10]),
                 k_m_ratio=0.1, dt_values=[0.1, 0.01, 0.001]):
    """Run complete analysis for both methods with different time steps"""
    analyzer = NumericalAnalysis()
    results = {}

    for dt in dt_values:
        results[dt] = {'euler': {}, 'rk4': {}}

        # Get reference solution for this dt
        t_ref = np.arange(t_span[0], t_span[1] + dt, dt)
        t_ref, states_ref = analyzer.get_reference_solution(t_span, initial_state, k_m_ratio, t_ref)

        for method in ['euler', 'rk4']:
            # Solve using numerical method
            t, states, runtime = analyzer.solve_numerical(
                method, t_span, initial_state, dt, k_m_ratio
            )

            # Calculate errors
            errors = analyzer.calculate_errors(states, states_ref)

            results[dt][method] = {
                'runtime': runtime,
                'errors': errors
            }

    return results

results = run_analysis(
    t_span=(0, 10),
    initial_state=np.array([0, 0, 10, 10]),  # [x, y, vx, vy]
    k_m_ratio=0.1,
    dt_values=[0.1, 0.01, 0.001]
)

# Print results for each time step and method
for dt in results:
    print(f"\nTime step: {dt}")
    for method in ['euler', 'rk4']:
        print(f"\n{method.upper()}:")
        print(f"Runtime: {results[dt][method]['runtime']:.6f} seconds")
        print("Errors:")
        for error_type, value in results[dt][method]['errors'].items():
            print(f"  {error_type}: {value:.2e}")


def plot_analysis_results(results):
    dt_values = list(results.keys())
    methods = ['euler', 'rk4']

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Numerical Methods: Euler vs RK4', fontsize=16)

    # Colors for consistent plotting
    colors = {'euler': 'blue', 'rk4': 'red'}

    # Runtime plot
    runtimes = {method: [results[dt][method]['runtime'] for dt in dt_values]
                for method in methods}

    ax1.loglog(dt_values, runtimes['euler'], 'o-', label='Euler', color=colors['euler'])
    ax1.loglog(dt_values, runtimes['rk4'], 'o-', label='RK4', color=colors['rk4'])
    ax1.set_xlabel('Time Step (dt)')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Runtime vs Time Step')
    ax1.grid(True)
    ax1.legend()

    # Maximum Absolute Error plot
    max_abs_errors = {method: [results[dt][method]['errors']['max_absolute_error']
                               for dt in dt_values] for method in methods}

    ax2.loglog(dt_values, max_abs_errors['euler'], 'o-', label='Euler', color=colors['euler'])
    ax2.loglog(dt_values, max_abs_errors['rk4'], 'o-', label='RK4', color=colors['rk4'])
    ax2.set_xlabel('Time Step (dt)')
    ax2.set_ylabel('Maximum Absolute Error')
    ax2.set_title('Maximum Absolute Error vs Time Step')
    ax2.grid(True)
    ax2.legend()

    # RMSE plot
    rmse = {method: [results[dt][method]['errors']['rmse']
                     for dt in dt_values] for method in methods}

    ax3.loglog(dt_values, rmse['euler'], 'o-', label='Euler', color=colors['euler'])
    ax3.loglog(dt_values, rmse['rk4'], 'o-', label='RK4', color=colors['rk4'])
    ax3.set_xlabel('Time Step (dt)')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Root Mean Square Error vs Time Step')
    ax3.grid(True)
    ax3.legend()

    # Maximum Relative Error plot
    max_rel_errors = {method: [results[dt][method]['errors']['max_relative_error']
                               for dt in dt_values] for method in methods}

    ax4.loglog(dt_values, max_rel_errors['euler'], 'o-', label='Euler', color=colors['euler'])
    ax4.loglog(dt_values, max_rel_errors['rk4'], 'o-', label='RK4', color=colors['rk4'])
    ax4.set_xlabel('Time Step (dt)')
    ax4.set_ylabel('Maximum Relative Error')
    ax4.set_title('Maximum Relative Error vs Time Step')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    return fig


# Convert your results to the right format
dt_values = [0.1, 0.01, 0.001]
results = {
    0.1: {
        'euler': {
            'runtime': 0.000611,
            'errors': {
                'max_absolute_error': 4.65e-01,
                'max_relative_error': 6.03e+00,
                'rmse': 1.73e-01
            }
        },
        'rk4': {
            'runtime': 0.002384,
            'errors': {
                'max_absolute_error': 1.05e-04,
                'max_relative_error': 3.58e-04,
                'rmse': 5.66e-05
            }
        }
    },
    0.01: {
        'euler': {
            'runtime': 0.006192,
            'errors': {
                'max_absolute_error': 4.36e-02,
                'max_relative_error': 1.96e+00,
                'rmse': 1.60e-02
            }
        },
        'rk4': {
            'runtime': 0.024533,
            'errors': {
                'max_absolute_error': 8.03e-09,
                'max_relative_error': 3.92e-07,
                'rmse': 4.25e-09
            }
        }
    },
    0.001: {
        'euler': {
            'runtime': 0.056412,
            'errors': {
                'max_absolute_error': 4.33e-03,
                'max_relative_error': 3.19e+00,
                'rmse': 1.59e-03
            }
        },
        'rk4': {
            'runtime': 0.235462,
            'errors': {
                'max_absolute_error': 1.72e-11,
                'max_relative_error': 2.41e-09,
                'rmse': 1.23e-12
            }
        }
    }
}

# Create the plots
fig = plot_analysis_results(results)
plt.show()