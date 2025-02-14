import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
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

    def rk4_integrate(self, times: np.ndarray, initial_state: np.ndarray, k_m_ratio: float) -> np.ndarray:
        """
        Improved RK4 integrator with adaptive timestep and enhanced stability checks
        """
        min_dt = 1e-4  # Minimum timestep
        max_dt = 0.1  # Maximum timestep
        tolerance = 1e-6  # Error tolerance

        states = [initial_state]
        current_time = times[0]
        dt = min((times[1] - times[0]), max_dt)

        while current_time < times[-1]:
            # Try two half steps
            half_dt = dt / 2
            k1 = self.equations_of_motion(current_time, states[-1], k_m_ratio)
            k2 = self.equations_of_motion(current_time + half_dt / 2, states[-1] + half_dt / 2 * k1, k_m_ratio)
            k3 = self.equations_of_motion(current_time + half_dt / 2, states[-1] + half_dt / 2 * k2, k_m_ratio)
            k4 = self.equations_of_motion(current_time + half_dt, states[-1] + half_dt * k3, k_m_ratio)
            first_half = states[-1] + (half_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Second half step
            k1 = self.equations_of_motion(current_time + half_dt, first_half, k_m_ratio)
            k2 = self.equations_of_motion(current_time + half_dt + half_dt / 2, first_half + half_dt / 2 * k1,
                                          k_m_ratio)
            k3 = self.equations_of_motion(current_time + half_dt + half_dt / 2, first_half + half_dt / 2 * k2,
                                          k_m_ratio)
            k4 = self.equations_of_motion(current_time + dt, first_half + half_dt * k3, k_m_ratio)
            two_half_steps = first_half + (half_dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Full step
            k1 = self.equations_of_motion(current_time, states[-1], k_m_ratio)
            k2 = self.equations_of_motion(current_time + dt / 2, states[-1] + dt / 2 * k1, k_m_ratio)
            k3 = self.equations_of_motion(current_time + dt / 2, states[-1] + dt / 2 * k2, k_m_ratio)
            k4 = self.equations_of_motion(current_time + dt, states[-1] + dt * k3, k_m_ratio)
            full_step = states[-1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Estimate error
            error = np.max(np.abs(two_half_steps - full_step))

            # Adjust timestep based on error
            if error > tolerance:
                dt = max(dt * 0.5, min_dt)
                continue

            # Accept step if we get here
            states.append(full_step)
            current_time += dt

            # Increase timestep if error is small
            if error < tolerance / 10:
                dt = min(dt * 1.5, max_dt)

            # Stability check
            if np.any(np.isnan(full_step)) or np.any(np.abs(full_step) > 1e10):
                self.logger.warning("Integration becoming unstable, stopping early")
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
        Generate predicted trajectory for given parameters using RK4
        """
        initial_state = np.array([
            self.positions[0][0] * self.pixel_to_meter,
            self.positions[0][1] * self.pixel_to_meter,
            vx0,
            vy0
        ])

        solution = self.rk4_integrate(times, initial_state, k_m_ratio)
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
        solution = self.rk4_integrate(adjusted_times, initial_state, self.trajectory_fit.k_m_ratio)
        return solution[:, :2]

    def analyze_video(self, video_path: str, output_path: Optional[str] = None) -> bool:
        """
        Analyze video to track ball and estimate parameters
        """
        self.video_path = video_path

        if not self.create_background_model(video_path):
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.dt = 1.0 / fps

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

                if out is not None:
                    out.write(frame)

                frame_count += 1

            cap.release()
            if out is not None:
                out.release()

            if len(self.positions) < 3:
                self.logger.error("Not enough ball positions detected")
                return False

            self.trajectory_fit = self.estimate_parameters()
            if self.trajectory_fit is None:
                return False

            t_flight = max(self.timestamps[-1] * 2.0, 2.0)
            total_frames = int(t_flight / self.dt)
            self.t_points = np.linspace(0, t_flight, total_frames)

            self.full_trajectory = self.generate_trajectory(
                self.trajectory_fit.k_m_ratio,
                self.trajectory_fit.initial_velocities[0],
                self.trajectory_fit.initial_velocities[1],
                self.t_points
            )

            print(f"Trajectory analysis completed:")
            print(f"- Number of frames: {total_frames}")
            print(f"- Time range: 0 to {t_flight:.2f} seconds")
            print(f"- Trajectory shape: {self.full_trajectory.shape}")

            return True

        except Exception as e:
            self.logger.error(f"Error analyzing video: {e}")
            return False

        finally:
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

    def create_animation(self, video_path: str, output_path: str = None, fps: int = 30):
        """
        Enhanced animation that includes both the original ball trajectory and a simulated shot
        """
        if not hasattr(self, 'full_trajectory'):
            raise ValueError("Must run analysis first - full_trajectory not found")
        if not hasattr(self, 'trajectory_fit'):
            raise ValueError("Must run analysis first - trajectory_fit not found")
        if not hasattr(self, 't_points'):
            raise ValueError("Must run analysis first - t_points not found")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            target_trajectory = self.full_trajectory
            print(f"Using stored trajectory with shape: {target_trajectory.shape}")

            shooter_pos = (0, frame_height * self.pixel_to_meter)

            shooting_solution = self.predict_interception(
                shooter_pos,
                target_trajectory,
                v0_initial=20
            )

            if shooting_solution is None:
                raise ValueError("Could not find valid shooting solution")

            angle, shooting_traj, shooting_delay = shooting_solution

            fig = plt.figure(figsize=(15, 6))
            gs = plt.GridSpec(1, 2)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title('Original Video')
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_title('Ball Trajectories')
            ax2.grid(True)
            ax2.set_xlabel('X Position (m)')
            ax2.set_ylabel('Y Position (m)')

            ret, first_frame = cap.read()
            if ret:
                im1 = ax1.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

            target_point = ax2.scatter([], [], c='blue', s=100, label='Target Ball')
            shooting_point = ax2.scatter([], [], c='red', s=100, label='Shooting Ball')
            target_line, = ax2.plot([], [], 'b--', alpha=0.5, label='Target Path')
            shooting_line, = ax2.plot([], [], 'r--', alpha=0.5, label='Shot Path')

            all_x = np.concatenate([target_trajectory[:, 0], shooting_traj[:, 0]])
            all_y = np.concatenate([target_trajectory[:, 1], shooting_traj[:, 1]])
            margin = 0.2
            x_range = np.ptp(all_x)
            y_range = np.ptp(all_y)
            ax2.set_xlim(np.min(all_x) - margin * x_range, np.max(all_x) + margin * x_range)
            ax2.set_ylim(np.min(all_y) - margin * y_range, np.max(all_y) + margin * y_range)

            ax2.legend()

            def update(frame):
                if frame < len(self.timestamps):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ret, video_frame = cap.read()
                    if ret:
                        im1.set_array(cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB))

                target_idx = min(frame, len(target_trajectory) - 1)
                target_pos = target_trajectory[target_idx]
                target_point.set_offsets(np.c_[target_pos[0], target_pos[1]])
                target_line.set_data(target_trajectory[:target_idx + 1, 0],
                                     target_trajectory[:target_idx + 1, 1])

                shooting_idx = frame - int(shooting_delay / self.dt)
                if 0 <= shooting_idx < len(shooting_traj):
                    shoot_pos = shooting_traj[shooting_idx]
                    shooting_point.set_offsets(np.c_[shoot_pos[0], shoot_pos[1]])
                    shooting_line.set_data(shooting_traj[:shooting_idx + 1, 0],
                                           shooting_traj[:shooting_idx + 1, 1])

                return [im1, target_point, shooting_point, target_line, shooting_line]

            total_frames = max(
                len(target_trajectory),
                int(shooting_delay / self.dt) + len(shooting_traj)
            )

            anim = animation.FuncAnimation(
                fig, update,
                frames=total_frames,
                interval=1000 / fps,
                blit=True
            )

            if output_path:
                if output_path.endswith('.gif'):
                    anim.save(output_path, writer='pillow', fps=fps)
                else:
                    anim.save(output_path, writer='ffmpeg', fps=fps)
                plt.close()

            return anim

        except Exception as e:
            self.logger.error(f"Error in create_animation: {e}")
            raise

        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()

    def rk4_step(self, t: float, state: np.ndarray, dt: float, k_m_ratio: float) -> np.ndarray:
        """
        Perform one step of RK4 integration.

        Args:
            t (float): Current time
            state (np.ndarray): Current state [x, y, vx, vy]
            dt (float): Time step
            k_m_ratio (float): Drag coefficient to mass ratio

        Returns:
            np.ndarray: New state after time step
        """
        # Calculate RK4 coefficients
        k1 = self.equations_of_motion(t, state, k_m_ratio)
        k2 = self.equations_of_motion(t + dt / 2, state + dt * k1 / 2, k_m_ratio)
        k3 = self.equations_of_motion(t + dt / 2, state + dt * k2 / 2, k_m_ratio)
        k4 = self.equations_of_motion(t + dt, state + dt * k3, k_m_ratio)

        # Update state using weighted sum
        return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def predict_interception(self, shooter_pos, target_trajectory, v0_initial=15):
        """
        Debug version of interception prediction with additional validation and logging
        """
        print(f"Starting interception prediction:")
        print(f"Shooter position: {shooter_pos}")
        print(f"Target trajectory length: {len(target_trajectory)}")
        print(f"Initial velocity: {v0_initial}")

        # Always use the stored full trajectory
        if not hasattr(self, 'full_trajectory'):
            raise ValueError("Must run analysis first")

        target_trajectory = self.full_trajectory
        t_points = self.t_points

        if not np.array_equal(target_trajectory, self.full_trajectory):
            print("Warning: Using different trajectory than analysis")

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
            # Update state using RK4
            state = self.rk4_step(t, state, self.dt, k_m_ratio)
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


if __name__ == "__main__":
    analyzer = EnhancedBallTrajectoryAnalyzer()
    video_path = "media/gio_test.mp4"
    output_path = "output_trajectory.avi"

    # Run analysis
    success = analyzer.analyze_video(video_path, output_path)
    if success:
        print("Analysis Results:")
        print(analyzer.get_results_summary())

        # Create enhanced animation with shooting simulation
        try:
            animation = analyzer.create_animation(
                video_path=video_path,
                output_path="animation_with_shot.gif",
                fps=30
            )
            plt.show()  # Display the animation
        except Exception as e:
            print(f"Error creating animation: {e}")




