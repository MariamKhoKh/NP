import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter

matplotlib.use('TkAgg')
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
import matplotlib.animation as animation


@dataclass
class TrajectoryFit:
    k_m_ratio: float
    initial_velocities: Tuple[float, float]
    r_squared: float
    rmse: float
    confidence_intervals: Optional[dict] = None


class EnhancedBallTrajectoryAnalyzer:
    def __init__(self, pixel_to_meter: float = 0.01,  dt: float = 1/30):
        self.positions: List[Tuple[int, int]] = []
        self.timestamps: List[float] = []
        self.background = None
        self.g = 9.81
        self.pixel_to_meter = pixel_to_meter
        self.trajectory_fit: Optional[TrajectoryFit] = None
        self.dt = dt  # time step between frames (default: 1/30 second)
        self.tolerance = 0.1  # tolerance for hit detection

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_background_model(self, video_path: str) -> bool:
        """
        Create a background model from random frames in the video
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Adjust number of samples based on video length
            n_samples = min(30, frame_count - 1)  # Ensure we don't try to sample more frames than available

            random_frames = np.random.choice(frame_count, n_samples, replace=False)
            frames = []

            for frame_idx in random_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

            cap.release()

            if frames:
                self.background = np.median(frames, axis=0).astype(np.uint8)
                return True

        except Exception as e:
            self.logger.error(f"Error creating background model: {e}")

        return False

    def detect_ball(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the ball in a single frame using background subtraction
        """
        if self.background is None:
            return None

        try:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(gray_frame, gray_background)
            blur = cv2.GaussianBlur(diff, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            best_match = None
            best_circularity = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                if circularity > best_circularity and circularity > 0.6:
                    best_circularity = circularity
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        best_match = (cx, cy)

            return best_match

        except Exception as e:
            self.logger.error(f"Error detecting ball: {e}")
            return None

    def equations_of_motion(self, t: float, state: np.ndarray, k_m_ratio: float) -> np.ndarray:
        """
        Implements the equations of motion for a ball with air resistance:
        dx/dt = vx
        dy/dt = vy
        dvx/dt = -(k/m)vx*sqrt(vx^2 + vy^2)
        dvy/dt = -g - (k/m)vy*sqrt(vx^2 + vy^2)
        """
        x, y, vx, vy = state

        # Calculate velocity magnitude
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)

        # Equations exactly as given in the mathematical model
        return np.array([
            vx,  # dx/dt
            vy,  # dy/dt
            -k_m_ratio * vx * v_magnitude,  # dvx/dt
            -self.g - k_m_ratio * vy * v_magnitude  # dvy/dt
        ])

    def trapezoid_integrate(self, times: np.ndarray, initial_state: np.ndarray, k_m_ratio: float) -> np.ndarray:
        """
        Trapezoidal method (Crank-Nicolson) integrator - A-stable method

        Args:
            times: Time points for integration
            initial_state: Initial state vector [x, y, vx, vy]
            k_m_ratio: Drag coefficient to mass ratio

        Returns:
            np.ndarray: Array of states at requested time points
        """
        # Integration parameters
        min_dt = 1e-4
        max_dt = 0.1
        tolerance = 1e-6
        max_iterations = 10

        # Initialize storage for states
        states = [initial_state]
        current_time = times[0]

        # Initial time step
        if len(times) > 1:
            dt = min(times[1] - times[0], max_dt)
        else:
            dt = max_dt

        # Main integration loop
        while current_time < times[-1] and len(states) < len(times) * 2:  # Safety limit
            current_state = states[-1]

            # Newton iteration for implicit step
            next_state = current_state.copy()
            converged = False

            for _ in range(max_iterations):
                # Evaluate derivatives at both ends
                f_n = self.equations_of_motion(current_time, current_state, k_m_ratio)
                f_np1 = self.equations_of_motion(current_time + dt, next_state, k_m_ratio)

                # Trapezoidal update formula
                next_state_new = current_state + (dt / 2) * (f_n + f_np1)

                # Check convergence
                error = np.max(np.abs(next_state_new - next_state))
                if error < tolerance:
                    next_state = next_state_new
                    converged = True
                    break

                next_state = next_state_new

            # Handle non-convergence
            if not converged:
                dt = max(dt * 0.5, min_dt)
                continue

            # Estimate local truncation error
            half_step = self.trapezoid_step(current_time, current_state, dt / 2, k_m_ratio)
            half_step = self.trapezoid_step(current_time + dt / 2, half_step, dt / 2, k_m_ratio)

            error_estimate = np.max(np.abs(next_state - half_step))

            # Adjust time step based on error
            if error_estimate > tolerance:
                dt = max(dt * 0.5, min_dt)
                continue

            if error_estimate < tolerance / 10 and dt < max_dt:
                dt = min(dt * 1.5, max_dt)

            # Accept the step if we get here
            states.append(next_state)
            current_time += dt

            # Check for instability
            if np.any(np.isnan(next_state)) or np.any(np.abs(next_state) > 1e10):
                break

        # Interpolate to get values at requested times
        times_computed = np.linspace(times[0], current_time, len(states))
        states_array = np.array(states)
        result = np.zeros((len(times), 4))

        for i in range(4):
            result[:, i] = np.interp(times, times_computed, states_array[:, i])

        return result

    def generate_trajectory(self, k_m_ratio: float, vx0: float, vy0: float,
                            times: np.ndarray) -> np.ndarray:
        """
        Generate predicted trajectory for given parameters using trapezoid
        """
        initial_state = np.array([
            self.positions[0][0] * self.pixel_to_meter,
            self.positions[0][1] * self.pixel_to_meter,
            vx0,
            vy0
        ])

        solution = self.trapezoid_integrate(times, initial_state, k_m_ratio)
        return solution[:, :2]

    def estimate_parameters(self) -> Optional[TrajectoryFit]:
        """
        Estimate trajectory parameters with improved coordinate system handling
        and velocity estimation

        Returns:
            Optional[TrajectoryFit]: Fitted trajectory parameters or None if estimation fails
        """
        try:
            # Get video dimensions
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Convert positions to meters and flip y-coordinates
            positions = np.array(self.positions) * self.pixel_to_meter
            positions[:, 1] = frame_height * self.pixel_to_meter - positions[:, 1]
            times = np.array(self.timestamps)

            # Improved velocity estimation using linear regression
            n_points = min(5, len(positions) - 1)
            t_fit = times[:n_points + 1].reshape(-1, 1)
            ones = np.ones_like(t_fit)

            # X velocity estimation
            X = np.hstack([ones, t_fit])
            vx0_guess = np.linalg.lstsq(X, positions[:n_points + 1, 0], rcond=None)[0][1]

            # Y velocity estimation
            vy0_guess = np.linalg.lstsq(X, positions[:n_points + 1, 1], rcond=None)[0][1]

            # Print debug information
            self.logger.info(f"Initial velocity estimates - vx0: {vx0_guess:.2f}, vy0: {vy0_guess:.2f}")

            # Set physically realistic bounds
            k_m_min, k_m_max = 0.001, 0.1  # Typical range for spherical objects
            v_margin = 1.1  # Allow 10% margin around initial velocity estimates

            # Ensure velocity bounds encompass the initial guesses
            vx_min = vx0_guess * (1 - v_margin) if vx0_guess > 0 else vx0_guess * (1 + v_margin)
            vx_max = vx0_guess * (1 + v_margin) if vx0_guess > 0 else vx0_guess * (1 - v_margin)
            vy_min = vy0_guess * (1 - v_margin) if vy0_guess > 0 else vy0_guess * (1 + v_margin)
            vy_max = vy0_guess * (1 + v_margin) if vy0_guess > 0 else vy0_guess * (1 - v_margin)

            bounds = ([k_m_min, vx_min, vy_min], [k_m_max, vx_max, vy_max])

            # Initial guess for k/m ratio based on typical ball properties
            k_m_guess = 0.02  # Starting point within typical range

            x0 = [
                k_m_guess,
                vx0_guess,
                vy0_guess
            ]

            def objective(params):
                k_m_ratio, vx0, vy0 = params
                predicted = self.generate_trajectory(k_m_ratio, vx0, vy0, times)
                if predicted is None or len(predicted) != len(positions):
                    return 1e10 * np.ones(2 * len(positions))

                # Weight the error terms to prioritize matching early trajectory points
                weights = np.exp(-2 * times / np.max(times))  # Increase decay rate
                weights = np.repeat(weights, 2)

                residuals = np.ravel(predicted - positions)
                return residuals * weights

            # Use robust least squares optimization
            result = least_squares(
                objective,
                x0=x0,
                bounds=bounds,
                method='trf',
                loss='soft_l1',
                max_nfev=500,
                ftol=1e-10,
                xtol=1e-10,
                verbose=2  # Add verbose output for debugging
            )

            if not result.success:
                raise RuntimeError(f"Parameter estimation failed to converge: {result.message}")

            # Generate predictions for metric calculations
            predictions = self.generate_trajectory(
                result.x[0], result.x[1], result.x[2], times
            )

            # Calculate goodness-of-fit metrics
            ss_res = np.sum((positions - predictions) ** 2)
            ss_tot = np.sum((positions - np.mean(positions, axis=0)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean((positions - predictions) ** 2))

            # Calculate confidence intervals using covariance matrix
            try:
                jac = result.jac
                cov = np.linalg.inv(jac.T @ jac) * (ss_res / (len(positions) - 3))
                confidence_intervals = {
                    'k_m': np.sqrt(cov[0, 0]) * 1.96,  # 95% confidence interval
                    'vx0': np.sqrt(cov[1, 1]) * 1.96,
                    'vy0': np.sqrt(cov[2, 2]) * 1.96
                }
            except np.linalg.LinAlgError:
                confidence_intervals = None
                self.logger.warning("Could not compute confidence intervals")

            # Create and return the trajectory fit
            return TrajectoryFit(
                k_m_ratio=result.x[0],
                initial_velocities=(result.x[1], result.x[2]),
                r_squared=r_squared,
                rmse=rmse,
                confidence_intervals=confidence_intervals
            )

        except Exception as e:
            self.logger.error(f"Error estimating parameters: {e}")
            return None

    def predict_full_trajectory(self, time_points: np.ndarray) -> np.ndarray:
        """
        Predict the complete trajectory using fitted parameters,
        ensuring smooth transition from observed to predicted path
        """
        if not self.trajectory_fit:
            raise ValueError("Must run analysis first")

        # Get the last observed state
        last_pos = np.array(self.positions[-1]) * self.pixel_to_meter

        # Get video dimensions for coordinate conversion
        cap = cv2.VideoCapture(self.video_path)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Flip last y coordinate to match physics convention
        last_pos[1] = frame_height * self.pixel_to_meter - last_pos[1]

        # Calculate velocities at the last observed point using finite differences
        if len(self.positions) >= 2:
            dt = self.timestamps[-1] - self.timestamps[-2]
            prev_pos = np.array(self.positions[-2]) * self.pixel_to_meter
            prev_pos[1] = frame_height * self.pixel_to_meter - prev_pos[1]

            vx = (last_pos[0] - prev_pos[0]) / dt
            vy = (last_pos[1] - prev_pos[1]) / dt
        else:
            # Fall back to fitted initial velocities if we don't have enough points
            vx, vy = self.trajectory_fit.initial_velocities

        # Create initial state vector for the continuation
        initial_state = np.array([
            last_pos[0],  # x position
            last_pos[1],  # y position
            vx,  # x velocity
            vy  # y velocity
        ])

        # Adjust time_points to start from the last observation
        last_t = self.timestamps[-1]
        adjusted_times = time_points - time_points[0] + last_t

        # Generate trajectory from the last observed state
        solution = self.trapezoid_integrate(adjusted_times, initial_state, self.trajectory_fit.k_m_ratio)
        return solution[:, :2]

    def analyze_video(self, video_path: str, output_path: Optional[str] = None) -> bool:
        """
        Analyze video to track ball and estimate parameters
        """
        self.video_path = video_path  # Store video path for later use

        if not self.create_background_model(video_path):
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Initialize video writer only if output path is provided
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(
                    filename=output_path,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=(frame_width, frame_height)
                )

            frame_count = 0
            self.positions = []
            self.timestamps = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ball_pos = self.detect_ball(frame)
                if ball_pos is not None:
                    self.positions.append(ball_pos)
                    self.timestamps.append(frame_count / fps)

                # Write frame to output video if writer exists
                if out is not None:
                    out.write(frame)

                frame_count += 1

            # Release resources
            cap.release()
            if out is not None:
                out.release()

            # Estimate parameters if we have enough positions
            if len(self.positions) < 3:
                self.logger.error("Not enough ball positions detected")
                return False

            self.trajectory_fit = self.estimate_parameters()
            if self.trajectory_fit is None:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error analyzing video: {e}")
            return False

        finally:
            # Ensure resources are released
            if 'cap' in locals() and cap is not None:
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()

    def get_results_summary(self) -> dict:
        """
        Get a summary of the analysis results
        """
        if not self.trajectory_fit:
            return {}

        return {
            'k_m_ratio': self.trajectory_fit.k_m_ratio,
            'vx, vy': self.trajectory_fit.initial_velocities,
            'r_squared': self.trajectory_fit.r_squared,
            'rmse': self.trajectory_fit.rmse,
            'number_of_points': len(self.positions),
            'trajectory_duration': self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0
        }

    def create_animation(self, video_path: str, output_path: str = None, fps: int = 30, shooting_data: dict = None):
        """
        Create enhanced animation with seamless trajectory integration and optional shooting visualization.

        Args:
            video_path: Path to the video file
            output_path: Path to save the animation
            fps: Frames per second
            shooting_data: Optional dictionary containing shooting information:
                - shooter_pos: (x, y) tuple of shooter position
                - trajectory: numpy array of shooting trajectory points
                - delay: shooting delay in seconds
        """
        if not self.trajectory_fit:
            raise ValueError("Must run analysis first")

        try:
            # Read video and get original FPS
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate frame duration in milliseconds
            interval = 1000 / original_fps

            # Read frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                raise ValueError("No frames read from video")

            # Convert frames to RGB
            frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

            # Convert observed positions to meters and flip y-coordinates
            observed = np.array(self.positions, dtype=np.float64) * self.pixel_to_meter
            observed[:, 1] = frame_height * self.pixel_to_meter - observed[:, 1]

            # Calculate time points for both observed and predicted trajectories
            last_observed_time = self.timestamps[-1]
            t_flight = max(last_observed_time * 2.0, 2.0)  # Extend prediction time

            # Create time points for the full trajectory
            total_frames = int(t_flight * original_fps)
            t_points = np.linspace(0, t_flight, total_frames)

            # Get full predicted trajectory
            full_trajectory = self.predict_full_trajectory(t_points)

            # Pre-compute velocities and accelerations using Savitzky-Golay filter
            window = 7  # Must be odd
            poly_order = 3

            if len(observed) < window:
                # If we have too few points, use a smaller window
                window = len(observed) if len(observed) % 2 == 1 else len(observed) - 1
                # Adjust poly_order if necessary (must be less than window size)
                poly_order = min(poly_order, window - 1)

            # Smooth the observed trajectories first
            smooth_observed = np.copy(observed)
            smooth_observed[:, 0] = savgol_filter(observed[:, 0], window, poly_order)
            smooth_observed[:, 1] = savgol_filter(observed[:, 1], window, poly_order)

            # Calculate velocities and accelerations using smoothed data
            dt = 1.0 / original_fps

            # Observed velocities and accelerations
            obs_velocities = np.zeros_like(smooth_observed)
            obs_accelerations = np.zeros_like(smooth_observed)

            for i in range(2):  # For both x and y components
                # Velocity using central differences of smoothed positions
                obs_velocities[1:-1, i] = (smooth_observed[2:, i] - smooth_observed[:-2, i]) / (2 * dt)
                obs_velocities[0, i] = obs_velocities[1, i]
                obs_velocities[-1, i] = obs_velocities[-2, i]

                # Acceleration using central differences of velocities
                obs_accelerations[1:-1, i] = (obs_velocities[2:, i] - obs_velocities[:-2, i]) / (2 * dt)
                obs_accelerations[0, i] = obs_accelerations[1, i]
                obs_accelerations[-1, i] = obs_accelerations[-2, i]

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 6))
            gs = plt.GridSpec(1, 2)

            # Original video subplot
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title('Original Video')

            # Trajectory subplot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_title('Trajectory Analysis')
            ax2.grid(True)
            ax2.set_xlabel('X Position (m)')
            ax2.set_ylabel('Y Position (m)')

            # Initialize plots
            im1 = ax1.imshow(frames_rgb[0])
            ax1.axis('off')

            # Initialize ball positions
            ball_scatter = ax2.scatter([], [], c='blue', s=100, label='Target Ball')

            # Initialize shooter visualization if shooting data is provided
            shooter_scatter = None
            if shooting_data:
                shooter_scatter = ax2.scatter([], [], c='red', s=100, label='Interceptor')
                ax2.scatter(
                    [shooting_data['shooter_pos'][0]],
                    [shooting_data['shooter_pos'][1]],
                    c='red',
                    marker='^',
                    s=100,
                    label='Shooter'
                )
                ax2.plot(
                    shooting_data['trajectory'][:, 0],
                    shooting_data['trajectory'][:, 1],
                    'r--',
                    alpha=0.5,
                    label='Interceptor Path'
                )

            # Plot observed points and predicted trajectory
            ax2.scatter(smooth_observed[:, 0], smooth_observed[:, 1],
                        color='gray', alpha=0.3, s=30, label='Observed')
            ax2.plot(full_trajectory[:, 0], full_trajectory[:, 1],
                     'b--', alpha=0.5, label='Predicted')

            # Set axis limits with margin
            all_points = [observed, full_trajectory]
            if shooting_data:
                all_points.append(shooting_data['trajectory'])
                all_points.append(np.array([shooting_data['shooter_pos']]))

            all_points = np.vstack(all_points)
            margin = 0.2
            x_range = np.ptp(all_points[:, 0])
            y_range = np.ptp(all_points[:, 1])
            ax2.set_xlim(np.min(all_points[:, 0]) - margin * x_range,
                         np.max(all_points[:, 0]) + margin * x_range)
            ax2.set_ylim(np.min(all_points[:, 1]) - margin * y_range,
                         np.max(all_points[:, 1]) + margin * y_range)

            ax2.legend()

            # Add metrics text box
            metrics_box = ax2.text(0.02, 0.98, '',
                                   transform=ax2.transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round',
                                             facecolor='white',
                                             alpha=0.8))

            # Define transition parameters
            transition_window = 20  # Increased for smoother transition
            transition_start = len(observed) - transition_window

            def physics_based_transition(current_frame):
                """
                Create physics-based smooth transition between observed and predicted trajectories
                using continuous jerk minimization.
                """
                if current_frame < transition_start:
                    return smooth_observed[current_frame], True

                elif current_frame < len(observed):
                    # Calculate normalized time in transition (0 to 1)
                    t = (current_frame - transition_start) / transition_window

                    # Use minimum jerk trajectory formula
                    # This ensures continuous position, velocity, and acceleration
                    # Formula: h(t) = 10t³ - 15t⁴ + 6t⁵
                    h = lambda t: t * t * t * (10 - t * (15 - 6 * t))

                    # Get start and end states
                    start_pos = smooth_observed[current_frame]
                    end_pos = full_trajectory[current_frame]
                    start_vel = obs_velocities[current_frame]
                    end_vel = np.zeros_like(start_vel)  # Assuming zero velocity at prediction point
                    start_acc = obs_accelerations[current_frame]
                    end_acc = np.zeros_like(start_acc)  # Assuming zero acceleration at prediction point

                    # Calculate blending coefficient
                    blend = h(t)

                    # Compute position using minimum jerk trajectory
                    pos = (1 - blend) * start_pos + blend * end_pos

                    # Add velocity and acceleration contributions with proper scaling
                    vel_contribution = ((1 - blend) * start_vel + blend * end_vel) * dt
                    acc_contribution = ((1 - blend) * start_acc + blend * end_acc) * dt * dt * 0.5

                    # Apply velocity and acceleration corrections with proper damping
                    damping = np.sin(np.pi * t)  # Smooth damping function
                    pos += vel_contribution * damping
                    pos += acc_contribution * damping * damping

                    return pos, True

                else:
                    if current_frame < len(full_trajectory):
                        return full_trajectory[current_frame], False
                    else:
                        return full_trajectory[-1], False

            def update(frame):
                # Update video frame
                if frame < len(frames_rgb):
                    im1.set_array(frames_rgb[frame])

                # Calculate current time
                current_time = frame * (1.0 / original_fps)

                # Get ball position using physics-based transition
                ball_pos, is_observed = physics_based_transition(frame)
                ball_scatter.set_offsets(np.c_[ball_pos[0], ball_pos[1]])

                # Update metrics text
                metrics_text = f'Time: {current_time:.2f}s'
                if is_observed:
                    predicted_pos = full_trajectory[frame] if frame < len(full_trajectory) else full_trajectory[-1]
                    error = np.linalg.norm(predicted_pos - ball_pos)
                    metrics_text += f'\nPrediction Error: {error:.2f}m'

                # Update shooter position if shooting data is provided
                if shooting_data and shooter_scatter is not None:
                    shooting_frame = frame - int(shooting_data['delay'] * original_fps)
                    if 0 <= shooting_frame < len(shooting_data['trajectory']):
                        shooter_pos = shooting_data['trajectory'][shooting_frame]
                        shooter_scatter.set_offsets(np.c_[shooter_pos[0], shooter_pos[1]])

                        # Check for collision
                        distance = np.linalg.norm(ball_pos - shooter_pos)
                        metrics_text += f'\nDistance: {distance:.2f}m'
                        if distance < 0.1:  # Collision threshold
                            metrics_text += '\nCOLLISION!'
                    else:
                        shooter_scatter.set_offsets(np.c_[[], []])

                metrics_box.set_text(metrics_text)

                return_elements = [im1, ball_scatter, metrics_box]
                if shooter_scatter is not None:
                    return_elements.append(shooter_scatter)
                return return_elements

            # Create animation
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=total_frames,
                interval=interval,
                blit=True
            )

            # Save animation if output path is provided
            if output_path:
                if output_path.endswith('.gif'):
                    writer = animation.PillowWriter(fps=original_fps)
                else:
                    writer = animation.FFMpegWriter(
                        fps=original_fps,
                        codec='h264',
                        bitrate=-1,
                        metadata=dict(title='Ball Trajectory Analysis')
                    )

                anim.save(output_path, writer=writer, dpi=100)
                plt.close()

            return anim

        except Exception as e:
            self.logger.error(f"Error creating animation: {e}")
            raise

    def trapezoid_step(self, t: float, state: np.ndarray, dt: float, k_m_ratio: float) -> np.ndarray:
        """
        Single step of trapezoidal method with improved stability

        Args:
            t: Current time
            state: Current state vector
            dt: Time step
            k_m_ratio: Drag coefficient to mass ratio

        Returns:
            np.ndarray: New state after time step
        """
        tolerance = 1e-6
        max_iterations = 10

        next_state = state.copy()
        f_n = self.equations_of_motion(t, state, k_m_ratio)

        # Newton iteration with improved convergence check
        for _ in range(max_iterations):
            f_np1 = self.equations_of_motion(t + dt, next_state, k_m_ratio)
            next_state_new = state + (dt / 2) * (f_n + f_np1)

            # Check convergence with relative tolerance
            rel_error = np.max(np.abs((next_state_new - next_state) /
                                      (np.abs(next_state) + tolerance)))

            if rel_error < tolerance:
                return next_state_new

            next_state = next_state_new

        # If we didn't converge, return the last iteration
        return next_state

    def predict_interception(self, shooter_pos, target_trajectory, v0_initial=15):
        """
        Debug version of interception prediction with additional validation and logging
        """
        print(f"Starting interception prediction:")
        print(f"Shooter position: {shooter_pos}")
        print(f"Target trajectory length: {len(target_trajectory)}")
        print(f"Initial velocity: {v0_initial}")

        # Validate inputs
        if len(target_trajectory) < 2:
            raise ValueError("Target trajectory too short")

        print(f"Target trajectory starts at: {target_trajectory[0]}")
        print(f"Target trajectory ends at: {target_trajectory[-1]}")

        target_times = np.arange(len(target_trajectory)) * self.dt
        best_solution = None
        min_error = float('inf')

        # Try shooting at different points in the target's trajectory
        for time_offset_idx in range(0, len(target_trajectory), 3):
            target_pos = target_trajectory[time_offset_idx]

            # Calculate distance and direction to target
            dx = target_pos[0] - shooter_pos[0]
            dy = target_pos[1] - shooter_pos[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)

            print(f"\nTrying to hit target at time {time_offset_idx * self.dt:.2f}s")
            print(f"Target position: {target_pos}")
            print(f"Distance to target: {distance:.2f}m")

            # Try different initial velocities
            for velocity_multiplier in np.linspace(0.5, 2.0, 10):
                v0 = v0_initial * velocity_multiplier

                # Calculate minimum angle needed to reach the distance
                min_angle = np.arcsin(self.g * distance / (v0 * v0)) / 2
                if np.isnan(min_angle):
                    continue

                print(f"\nTrying velocity: {v0:.2f} m/s")
                print(f"Minimum angle needed: {np.degrees(min_angle):.2f} degrees")

                # Search around the minimum angle
                base_angle = np.arctan2(dy, dx)
                for angle_offset in np.linspace(-np.pi / 4, np.pi / 4, 20):
                    angle = base_angle + angle_offset

                    # Calculate trajectory
                    shooting_traj = self.calculate_trajectory(
                        shooter_pos[0],
                        shooter_pos[1],
                        v0,
                        angle,
                        target_pos[0],
                        target_pos[1],
                        k_m_ratio=0.02,  # Lower air resistance
                        max_time=3.0
                    )

                    if shooting_traj is None:
                        continue

                    # Check for intersection
                    for i, shoot_pos in enumerate(shooting_traj):
                        shoot_time = i * self.dt
                        target_idx = int((shoot_time + time_offset_idx * self.dt) / self.dt)

                        if target_idx >= len(target_trajectory):
                            break

                        target_pos = target_trajectory[target_idx]
                        distance = np.sqrt(np.sum((shoot_pos - target_pos) ** 2))

                        if distance <= self.tolerance * 3:  # Increased tolerance
                            print(f"Found potential solution!")
                            print(f"Impact at time: {shoot_time:.2f}s")
                            print(f"Distance to target at impact: {distance:.2f}m")

                            if distance < min_error:
                                min_error = distance
                                best_solution = (angle, shooting_traj, time_offset_idx * self.dt)
                                print(f"New best solution found with error: {min_error:.2f}m")
                            break

        if best_solution is None:
            print("No valid solution found!")

            # Add some diagnostic information about the target trajectory
            height_range = np.ptp(target_trajectory[:, 1])
            distance_range = np.ptp(target_trajectory[:, 0])
            print(f"Target trajectory height range: {height_range:.2f}m")
            print(f"Target trajectory distance range: {distance_range:.2f}m")
            print(f"Target velocity (approximate): {np.mean(np.diff(target_trajectory[:, 0])) / self.dt:.2f}m/s")

            raise ValueError("Could not find valid shooting solution")

        print(f"\nFinal solution found:")
        print(f"Launch angle: {np.degrees(best_solution[0]):.2f} degrees")
        print(f"Shooting delay: {best_solution[2]:.2f}s")
        print(f"Final error: {min_error:.2f}m")

        return best_solution

    def shoot_at_moving_target(self, shooter_pos, target_trajectory, animate=True):
        """
        Calculate and visualize shooting at a moving target

        Args:
            shooter_pos: Tuple[float, float] - Starting position for the shot
            target_trajectory: np.ndarray - Precomputed target trajectory
            animate: bool - Whether to create an animation
        """
        solution = self.predict_interception(shooter_pos, target_trajectory)
        if solution is None:
            raise ValueError("Could not find a valid shooting solution")

        angle, shooting_traj, shooting_delay = solution

        if not animate:
            return angle, shooting_traj, shooting_delay

        # Create animation
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, max(np.max(target_trajectory[:, 0]), np.max(shooting_traj[:, 0])) * 1.1)
        ax.set_ylim(0, max(np.max(target_trajectory[:, 1]), np.max(shooting_traj[:, 1])) * 1.1)
        ax.grid(True)

        # Initialize plot elements
        target, = ax.plot([], [], 'bo', markersize=10, label='Target')
        projectile, = ax.plot([], [], 'ro', markersize=8, label='Projectile')
        target_trail, = ax.plot([], [], 'b--', alpha=0.3)
        shooting_trail, = ax.plot([], [], 'r--', alpha=0.3)

        ax.legend()

        def init():
            target.set_data([], [])
            projectile.set_data([], [])
            target_trail.set_data([], [])
            shooting_trail.set_data([], [])
            return target, projectile, target_trail, shooting_trail

        def animate(frame):
            # Update target position
            target_idx = frame
            target_pos = target_trajectory[min(target_idx, len(target_trajectory) - 1)]
            target.set_data([target_pos[0]], [target_pos[1]])
            target_trail.set_data(
                target_trajectory[:target_idx + 1, 0],
                target_trajectory[:target_idx + 1, 1]
            )

            # Update projectile position if it's time to shoot
            shooting_idx = frame - int(shooting_delay / self.dt)
            if shooting_idx >= 0 and shooting_idx < len(shooting_traj):
                proj_pos = shooting_traj[shooting_idx]
                projectile.set_data([proj_pos[0]], [proj_pos[1]])
                shooting_trail.set_data(
                    shooting_traj[:shooting_idx + 1, 0],
                    shooting_traj[:shooting_idx + 1, 1]
                )
            else:
                projectile.set_data([], [])
                shooting_trail.set_data([], [])

            return target, projectile, target_trail, shooting_trail

        total_frames = max(
            len(target_trajectory),
            int(shooting_delay / self.dt) + len(shooting_traj)
        )

        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=total_frames, interval=20, blit=True
        )

        return anim

    def calculate_trajectory(self, x0, y0, v0, angle, target_x, target_y, k_m_ratio=0.02, max_time=3.0):
        """
        Modified trajectory calculation with better debugging
        """
        # Initial velocities
        vx0 = v0 * np.cos(angle)
        vy0 = v0 * np.sin(angle)

        # Initial state vector [x, y, vx, vy]
        state = np.array([x0, y0, vx0, vy0])

        # Initialize lists to store trajectory
        trajectory = [state[:2]]  # Store only positions
        t = 0

        while t < max_time:
            # Update state using trapezoid
            state = self.trapezoid_step(t, state, self.dt, k_m_ratio)
            t += self.dt

            # Store position
            trajectory.append(state[:2])

            # Stop if we hit the ground
            if state[1] < 0:
                break

            # Stop if we've gone too far
            if abs(state[0] - x0) > 100 or abs(state[1] - y0) > 100:
                break

        trajectory = np.array(trajectory)

        # Validate trajectory
        if len(trajectory) < 2:
            return None

        return trajectory

    def shooting_method_interception(self, shooter_pos, target_trajectory, v0_range=(10.0, 30.0),
                                     angle_range=(-np.pi / 3, np.pi / 3), max_iterations=100, tolerance=0.1):
        """
        Enhanced shooting method for target interception with multiple initial velocity attempts
        and better convergence properties.

        Args:
            shooter_pos: (x, y) position of the shooter
            target_trajectory: Array of target positions over time
            v0_range: Tuple of (min, max) initial velocities to try
            angle_range: Tuple of (min, max) angles to try
            max_iterations: Maximum iterations for each attempt
            tolerance: Acceptable error for convergence

        Returns:
            tuple: (optimal_angle, optimal_v0, shooting_trajectory, shooting_delay)
        """

        def objective_function(params):
            """Calculate minimum distance between trajectories with penalties."""
            angle, v0, delay = params

            # Calculate shooter trajectory
            shooting_traj = self.calculate_trajectory(
                shooter_pos[0], shooter_pos[1],
                v0, angle, 0, 0,
                k_m_ratio=0.02
            )

            if shooting_traj is None or len(shooting_traj) < 2:
                return float('inf')

            # Time points
            shooter_times = np.arange(len(shooting_traj)) * self.dt
            delay_idx = int(delay / self.dt)

            # Calculate minimum distance with time penalty
            min_distance = float('inf')
            intercept_height = -float('inf')
            for i, shoot_pos in enumerate(shooting_traj):
                target_idx = delay_idx + i
                if target_idx >= len(target_trajectory):
                    break

                target_pos = target_trajectory[target_idx]
                distance = np.linalg.norm(shoot_pos - target_pos)

                # Add penalty for low heights to prevent ground solutions
                height_penalty = max(0, 1.0 - shoot_pos[1])

                # Add penalty for very long trajectories
                time_penalty = 0.1 * i * self.dt

                total_distance = distance + height_penalty + time_penalty

                if total_distance < min_distance:
                    min_distance = total_distance
                    intercept_height = shoot_pos[1]

            # Add penalty for solutions too close to target trajectory
            min_dist_to_target_path = float('inf')
            for target_pos in target_trajectory:
                dist = np.linalg.norm(np.array([shooter_pos[0], shooter_pos[1]]) - target_pos)
                min_dist_to_target_path = min(min_dist_to_target_path, dist)

            proximity_penalty = max(0, 5.0 - min_dist_to_target_path)

            return min_distance + proximity_penalty

        best_solution = None
        best_error = float('inf')

        # Try different initial velocities
        v0_values = np.linspace(v0_range[0], v0_range[1], 5)

        for v0_guess in v0_values:
            try:
                # Initial guess for parameters [angle, v0, delay]
                dx = target_trajectory[0][0] - shooter_pos[0]
                dy = target_trajectory[0][1] - shooter_pos[1]
                initial_angle = np.clip(np.arctan2(dy, dx), angle_range[0], angle_range[1])
                params = np.array([initial_angle, v0_guess, 0.0])

                # Gradient descent with momentum
                velocity = np.zeros_like(params)
                momentum = 0.9
                learning_rate = 0.01

                for iteration in range(max_iterations):
                    error = objective_function(params)

                    if error < tolerance:
                        break

                    # Calculate numerical gradient
                    gradient = np.zeros_like(params)
                    eps = 1e-6
                    for i in range(len(params)):
                        params_plus = params.copy()
                        params_plus[i] += eps
                        params_minus = params.copy()
                        params_minus[i] -= eps
                        gradient[i] = (objective_function(params_plus) - objective_function(params_minus)) / (2 * eps)

                    # Update with momentum
                    velocity = momentum * velocity - learning_rate * gradient
                    params += velocity

                    # Apply constraints
                    params[0] = np.clip(params[0], angle_range[0], angle_range[1])  # angle
                    params[1] = np.clip(params[1], v0_range[0], v0_range[1])  # velocity
                    params[2] = max(0.0, params[2])  # delay

                    if error < best_error:
                        best_error = error
                        best_solution = params.copy()

            except Exception as e:
                self.logger.warning(f"Attempt with v0={v0_guess} failed: {str(e)}")
                continue

        if best_solution is None:
            raise ValueError("Could not find valid solution with any configuration")

        optimal_angle, optimal_v0, optimal_delay = best_solution

        # Calculate final trajectory with optimal parameters
        optimal_trajectory = self.calculate_trajectory(
            shooter_pos[0], shooter_pos[1],
            optimal_v0, optimal_angle, 0, 0,
            k_m_ratio=0.02
        )

        return optimal_angle, optimal_v0, optimal_trajectory, optimal_delay


if __name__ == "__main__":
    analyzer = EnhancedBallTrajectoryAnalyzer()
    video_path = "media/blue_ball.mp4"
    output_path = "output_trajectory.avi"

    # Run analysis
    success = analyzer.analyze_video(video_path, output_path)
    if success:
        print("Analysis Results:")
        print(analyzer.get_results_summary())

        try:
            # Calculate full trajectory for the target ball
            t_points = np.linspace(0, analyzer.timestamps[-1] * 2.0,
                                   int(analyzer.timestamps[-1] * 2.0 * 30))
            full_trajectory = analyzer.predict_full_trajectory(t_points)

            # Print trajectory information
            print("\nTrajectory Information:")
            print(f"Duration: {analyzer.timestamps[-1]:.2f} seconds")
            print(f"Number of points: {len(full_trajectory)}")
            print(f"Start position: ({full_trajectory[0][0]:.2f}, {full_trajectory[0][1]:.2f})")
            print(f"End position: ({full_trajectory[-1][0]:.2f}, {full_trajectory[-1][1]:.2f})")

            # Define search space for shooter positions
            # Positions are now defined relative to the starting point of the ball
            ball_start = full_trajectory[0]
            x_offset = -2.0  # meters to the left of the ball's path
            y_offset = 0.5  # meters above ground

            shooter_positions = [
                (ball_start[0] + x_offset, y_offset),
                (ball_start[0] + x_offset - 1.0, y_offset),
                (ball_start[0] + x_offset - 2.0, y_offset),
                (ball_start[0] + x_offset, y_offset + 0.5)
            ]

            # Try different shooter positions with improved shooting method
            success = False
            best_error = float('inf')
            best_solution = None

            for pos in shooter_positions:
                print(f"\nTrying shooter position: {pos}")
                try:
                    # Use the improved shooting method with velocity range
                    angle, v0, shooting_trajectory, shooting_delay = analyzer.shooting_method_interception(
                        pos,
                        full_trajectory,
                        v0_range=(15.0, 30.0),
                        angle_range=(-np.pi / 3, np.pi / 3)
                    )

                    # Calculate error metric for this solution
                    min_distance = float('inf')
                    for i, shoot_pos in enumerate(shooting_trajectory):
                        target_idx = int(shooting_delay / analyzer.dt) + i
                        if target_idx < len(full_trajectory):
                            distance = np.linalg.norm(shoot_pos - full_trajectory[target_idx])
                            min_distance = min(min_distance, distance)

                    if min_distance < best_error:
                        best_error = min_distance
                        best_solution = {
                            'shooter_pos': pos,
                            'angle': angle,
                            'v0': v0,
                            'trajectory': shooting_trajectory,
                            'delay': shooting_delay
                        }
                        success = True

                    print(f"Found solution with error: {min_distance:.2f}m")
                    print(f"Launch angle: {np.degrees(angle):.2f} degrees")
                    print(f"Initial velocity: {v0:.2f} m/s")
                    print(f"Shooting delay: {shooting_delay:.2f} seconds")

                except ValueError as e:
                    print(f"Failed with this configuration: {e}")
                    continue

            if not success:
                print("\nCould not find valid shooting solution with any configuration")
                print("Creating animation without shooting visualization...")
                animation = analyzer.create_animation(
                    video_path=video_path,
                    output_path="animation_no_shooting.gif",
                    fps=30
                )
            else:
                print("\nCreating animation with best shooting solution...")
                print(f"Best solution found:")
                print(f"Shooter position: {best_solution['shooter_pos']}")
                print(f"Launch angle: {np.degrees(best_solution['angle']):.2f} degrees")
                print(f"Initial velocity: {best_solution['v0']:.2f} m/s")
                print(f"Shooting delay: {best_solution['delay']:.2f} seconds")
                print(f"Minimum error: {best_error:.2f}m")

                animation = analyzer.create_animation(
                    video_path=video_path,
                    output_path="stability.gif",
                    fps=30,
                    shooting_data={
                        'shooter_pos': best_solution['shooter_pos'],
                        'trajectory': best_solution['trajectory'],
                        'delay': best_solution['delay']
                    }
                )

            plt.show()

        except Exception as e:
            print(f"Error creating animation: {e}")
            print("Stack trace:")
            import traceback

            traceback.print_exc()
