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
    def __init__(self, pixel_to_meter: float = 0.01):
        self.positions: List[Tuple[int, int]] = []
        self.timestamps: List[float] = []
        self.background = None
        self.g = 9.81
        self.pixel_to_meter = pixel_to_meter
        self.trajectory_fit: Optional[TrajectoryFit] = None

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

    def create_animation(self, video_path: str, output_path: str = None, fps: int = 30):
        """
        Create enhanced animation with seamless trajectory integration,
        showing ball movement through both observed and predicted paths.
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

            # Calculate full trajectory including prediction
            t_flight = max(self.timestamps[-1] * 2.0, 2.0)  # Extend prediction time
            t_points = np.linspace(0, t_flight, int(t_flight * original_fps))

            # Get full predicted trajectory
            full_trajectory = self.predict_full_trajectory(t_points)

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 8))
            gs = plt.GridSpec(2, 2, height_ratios=[3, 1])

            # Original video subplot
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title('Original Video')

            # Trajectory subplot
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.set_title('Trajectory Analysis')
            ax2.grid(True)
            ax2.set_xlabel('X Position (m)')
            ax2.set_ylabel('Y Position (m)')

            # Validation metrics subplot
            ax3 = fig.add_subplot(gs[1, :])
            ax3.set_title('Validation Metrics')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Error (m)')
            ax3.grid(True)

            # Initialize plots
            im1 = ax1.imshow(frames_rgb[0])
            ax1.axis('off')

            # Initialize ball position scatter
            ball_scatter = ax2.scatter([], [], c='blue', s=100)

            # Plot observed points
            ax2.scatter(observed[:, 0], observed[:, 1], color='red', alpha=0.3, s=30, label='Observed')

            # Plot complete predicted trajectory
            ax2.plot(full_trajectory[:, 0], full_trajectory[:, 1], 'g--', alpha=0.5, label='Predicted')

            # Set axis limits with margin
            all_x = np.concatenate([observed[:, 0], full_trajectory[:, 0]])
            all_y = np.concatenate([observed[:, 1], full_trajectory[:, 1]])
            margin = 0.2
            x_range = np.ptp(all_x)
            y_range = np.ptp(all_y)
            ax2.set_xlim(np.min(all_x) - margin * x_range, np.max(all_x) + margin * x_range)
            ax2.set_ylim(np.min(all_y) - margin * y_range, np.max(all_y) + margin * y_range)

            ax2.legend()

            # Initialize error plot
            error_line, = ax3.plot([], [], 'r-', label='Prediction Error')
            ax3.legend()
            ax3.set_xlim(0, t_flight)
            ax3.set_ylim(0, 1)  # Will be adjusted during animation

            # Add metrics text box
            metrics_box = ax2.text(0.02, 0.98, '',
                                   transform=ax2.transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round',
                                             facecolor='white',
                                             alpha=0.8))

            def update(frame):
                # Update video frame
                if frame < len(frames_rgb):
                    im1.set_array(frames_rgb[frame])

                current_time = frame / original_fps

                # Key change: Use observed positions when available, otherwise use predicted
                if frame < len(observed):
                    # Use observed position
                    ball_pos = observed[frame]
                    point_color = 'blue'
                else:
                    # Use predicted position
                    ball_pos = full_trajectory[frame]
                    point_color = 'blue'

                # Update ball position and color
                ball_scatter.set_offsets(np.c_[ball_pos[0], ball_pos[1]])
                ball_scatter.set_color(point_color)

                # Update metrics
                if frame < len(self.timestamps):
                    if frame < len(observed):
                        predicted_pos = full_trajectory[frame]
                        actual_pos = observed[frame]
                        error = np.linalg.norm(predicted_pos - actual_pos)

                        times = self.timestamps[:frame + 1]
                        errors = [np.linalg.norm(full_trajectory[i] - observed[i])
                                  for i in range(frame + 1)]

                        error_line.set_data(times, errors)

                        if errors:
                            max_error = max(errors)
                            if max_error > 0:
                                ax3.set_ylim(0, max_error * 1.2)

                        metrics_text = (f'Time: {current_time:.2f}s\n'
                                        f'Current Error: {error:.2f}m\n'
                                        f'Mean Error: {np.mean(errors):.2f}m\n'
                                        f'Max Error: {max_error:.2f}m')
                    else:
                        metrics_text = f'Time: {current_time:.2f}s\nPredicted Position'

                    metrics_box.set_text(metrics_text)

                return [im1, ball_scatter, error_line, metrics_box]

            # Create animation
            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(full_trajectory),  # Use full trajectory length
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



if __name__ == "__main__":
    analyzer = EnhancedBallTrajectoryAnalyzer()
    video_path = "media/origin_cut.mp4"
    output_path = "output_trajectory.avi"  # Use .avi extension for XVID codec

    # Run analysis
    success = analyzer.analyze_video(video_path, output_path)
    if success:
        print("Analysis Results:")
        print(analyzer.get_results_summary())

        # Create animation separately
        try:
            analyzer.create_animation(video_path, "animation.gif", fps=30)
        except Exception as e:
            print(f"Error creating animation: {e}")


