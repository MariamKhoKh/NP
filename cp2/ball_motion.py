import numpy as np
import cv2
from scipy.optimize import least_squares
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging


@dataclass
class TrajectoryFit:
    """Stores the results of trajectory fitting"""
    k_m_ratio: float
    initial_velocities: Tuple[float, float]
    r_squared: float
    rmse: float
    confidence_intervals: Optional[np.ndarray] = None


class BallTrajectoryAnalyzer:
    def __init__(self, pixel_to_meter: float = 0.01):
        """
        Initialize the ball trajectory analyzer

        Args:
            pixel_to_meter: Conversion factor from pixels to meters
        """
        self.positions: List[Tuple[int, int]] = []
        self.timestamps: List[float] = []
        self.background = None
        self.g = 9.81  # acceleration due to gravity (m/s²)
        self.pixel_to_meter = pixel_to_meter
        self.trajectory_fit: Optional[TrajectoryFit] = None

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def equations_of_motion(t: float, state: np.ndarray, k_m_ratio: float) -> np.ndarray:
        """
        Implement the equations of motion for the ball

        Args:
            t: Time (not used, but required for solve_ivp)
            state: Array [x, y, vx, vy]
            k_m_ratio: k/m ratio for air resistance

        Returns:
            Array of derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt]
        """
        _, _, vx, vy = state
        v_magnitude = np.sqrt(vx ** 2 + vy ** 2)

        return np.array([
            vx,  # dx/dt
            vy,  # dy/dt
            -k_m_ratio * vx * v_magnitude,  # dvx/dt
            -9.81 - k_m_ratio * vy * v_magnitude  # dvy/dt
        ])

    def create_background_model(self, video_path: str) -> bool:
        """
        Create a background model from random frames in the video

        Args:
            video_path: Path to the video file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            random_frames = np.random.choice(frame_count, 30, replace=False)
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
        and contour analysis

        Args:
            frame: Input video frame

        Returns:
            Tuple of (x, y) coordinates or None if ball not detected
        """
        if self.background is None:
            return None

        try:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)

            # Background subtraction
            diff = cv2.absdiff(gray_frame, gray_background)
            blur = cv2.GaussianBlur(diff, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Find most circular contour
            best_match = None
            best_circularity = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Minimum area threshold
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

    def estimate_parameters(self) -> Optional[TrajectoryFit]:
        """
        Estimate k/m ratio and initial velocities from the trajectory data
        """
        try:

            positions = np.array(self.positions) * self.pixel_to_meter
            times = np.array(self.timestamps)

            def objective(params):
                """Objective function for least squares optimization"""
                k_m_ratio, vx0, vy0 = params
                predicted_positions = self.generate_trajectory(k_m_ratio, vx0, vy0, times)
                return np.ravel(predicted_positions - positions)

            # Initial guess based on finite differences
            vx0_guess = (positions[1, 0] - positions[0, 0]) / (times[1] - times[0])
            vy0_guess = (positions[1, 1] - positions[0, 1]) / (times[1] - times[0])

            # Bounds for parameters
            bounds = ([0, -100, -100],  # lower bounds for k/m, vx0, vy0
                      [1, 100, 100])  # upper bounds for k/m, vx0, vy0

            # Try different combinations of k/m, vx0, vy0 to find the best match
            result = least_squares(
                objective,
                x0=[0.1, vx0_guess, vy0_guess],  # starting guess
                bounds=bounds,
                method='trf',
                loss='soft_l1'  # More robust to outliers
            )

            if not result.success:
                raise RuntimeError("Parameter estimation failed to converge")

            # Calculate fit quality metrics
            predictions = self.generate_trajectory(result.x[0], result.x[1],
                                                   result.x[2], times)
            r_squared = 0.0
            rmse = 0.0

            # Placeholder since estimate_confidence_intervals was removed
            confidence = None

            return TrajectoryFit(
                k_m_ratio=result.x[0],
                initial_velocities=(result.x[1], result.x[2]),
                r_squared=r_squared,
                rmse=rmse,
                confidence_intervals=confidence
            )

        except Exception as e:
            self.logger.error(f"Error estimating parameters: {e}")
            return None

    def rk4_integrate(self, times: np.ndarray, initial_state: np.ndarray,
                      k_m_ratio: float) -> np.ndarray:
        """
        Custom RK4 integrator for the equations of motion

        Args:
            times: Time points to evaluate
            initial_state: Initial conditions [x0, y0, vx0, vy0]
            k_m_ratio: k/m ratio parameter

        Returns:
            Array of states at specified times
        """
        dt = times[1] - times[0]
        n_steps = len(times)
        states = np.zeros((n_steps, 4))
        states[0] = initial_state

        for i in range(1, n_steps):
            # RK4 steps
            k1 = self.equations_of_motion(times[i - 1], states[i - 1], k_m_ratio)
            k2 = self.equations_of_motion(times[i - 1] + dt / 2,
                                          states[i - 1] + dt * k1 / 2, k_m_ratio)
            k3 = self.equations_of_motion(times[i - 1] + dt / 2,
                                          states[i - 1] + dt * k2 / 2, k_m_ratio)
            k4 = self.equations_of_motion(times[i - 1] + dt,
                                          states[i - 1] + dt * k3, k_m_ratio)

            states[i] = states[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return states

    def generate_trajectory(self, k_m_ratio: float, vx0: float, vy0: float,
                            times: np.ndarray) -> np.ndarray:
        """
        Generate predicted trajectory for given parameters using RK4

        Args:
            k_m_ratio: Drag coefficient / mass ratio
            vx0, vy0: Initial velocities
            times: Time points to evaluate

        Returns:
            Array of predicted positions
        """
        initial_state = np.array([
            self.positions[0][0] * self.pixel_to_meter,
            self.positions[0][1] * self.pixel_to_meter,
            vx0,
            vy0
        ])

        # Use custom RK4 integrator
        solution = self.rk4_integrate(times, initial_state, k_m_ratio)

        return solution[:, :2]  # Return only position components

    def analyze_video(self, video_path: str, output_path: Optional[str] = None) -> bool:
        """
        Analyze video to track ball and estimate parameters

        Args:
            video_path: Path to input video
            output_path: Optional path for annotated output video

        Returns:
            bool: True if analysis was successful
        """
        if not self.create_background_model(video_path):
            return False

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file")

            # Video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Initialize video writer if output path provided
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps,
                                      (frame_width, frame_height))

            # First pass: Detect ball positions
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

                frame_count += 1

            # Estimate parameters
            self.trajectory_fit = self.estimate_parameters()
            if self.trajectory_fit is None:
                return False

            # Second pass: Annotate video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw detected ball position
                if frame_count < len(self.positions):
                    cv2.circle(frame, self.positions[frame_count], 5, (0, 255, 0), -1)

                # Draw trajectory
                if len(self.positions) > 1:
                    for i in range(1, len(self.positions)):
                        pt1 = self.positions[i - 1]
                        pt2 = self.positions[i]
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

                # Add parameter information
                if self.trajectory_fit:
                    text_lines = [
                        f"k/m ratio: {self.trajectory_fit.k_m_ratio:.4e}",
                        f"velocity: {np.hypot(*self.trajectory_fit.initial_velocities):.2f} m/s",]

                    for i, line in enumerate(text_lines):
                        cv2.putText(frame, line, (10, 30 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if output_path and out is not None:
                        out.write(frame)

                        cv2.imshow('Ball Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1

            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()

            return True

        except Exception as e:
            self.logger.error(f"Error analyzing video: {e}")
            return False

    def get_results_summary(self) -> dict:
        """
        Get a summary of the analysis results

        Returns:
            Dictionary containing analysis results
        """
        if not self.trajectory_fit:
            return {}

        return {
            'k_m_ratio': self.trajectory_fit.k_m_ratio,
            'vx, vy': self.trajectory_fit.initial_velocities,
            'r_squared': self.trajectory_fit.r_squared,
            'number_of_points': len(self.positions),
            'trajectory_duration': self.timestamps[-1] - self.timestamps[0] if self.timestamps else 0
        }


def main():
    """Example usage of the BallTrajectoryAnalyzer"""
    pixel_to_meter = 0.01  # assumption: 1 pixel = 1 cm since we do not know exact distance from point A to point B
    analyzer = BallTrajectoryAnalyzer(pixel_to_meter=pixel_to_meter)

    video_path = "canva.mp4"
    output_path = "analyzed_trajectory.avi"

    if analyzer.analyze_video(video_path, output_path):
        results = analyzer.get_results_summary()
        print("\nAnalysis Results:")
        print("-" * 40)
        print(f"k/m ratio: {results['k_m_ratio']:.4e}")
        print(f"Initial velocity (x, y): ({results['vx, vy'][0]:.2f}, "
              f"{results['vx, vy'][1]:.2f}) m/s")
        print(f"Number of tracked points: {results['number_of_points']}")
        print(f"Trajectory duration: {results['trajectory_duration']:.2f} s")


    else:
        print("Analysis failed. Check the logs for details.")


if __name__ == "__main__":
    main()
