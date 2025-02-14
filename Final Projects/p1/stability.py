import cv2
import numpy as np
import matplotlib
from scipy.optimize import fsolve

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from scipy.ndimage import convolve

"""
The Trapezoidal method, while A-stable, requires solving an implicit equation at each step, 
which is computationally more expensive than explicit methods like RK4.
With larger image sizes, it's likely processing more pixels and potentially detecting more balls, 
which multiplies the computational cost.

So, it's preferable to test the following program for the images with resolution 
1080x720(aka HD or 720p, has total of 921,600 pixels)
or less 
"""

class BallShooting:
    def __init__(self):
        self.g = 9.81  # gravity
        self.dt = 0.01  # time step
        self.tolerance = 5.0  # hit detection tolerance
        self.k_m_ratio = 0.001  # air resistance coefficient / mass ratio
        self.trajectory_cache = {}  # cache for trajectories
        self.initial_height = None  # will be set when loading image
        self.trajectory_line = None  # will be set in setup_plots
        self.moving_ball = None  # will be set in setup_plots

    @staticmethod
    def manual_canny(image, sigma=1.0, low_threshold=None, high_threshold=None):
        """
        Manual implementation of Canny edge detector

        Parameters:
        - image: Input grayscale image (2D numpy array)
        - sigma: Standard deviation for Gaussian filter
        - low_threshold: Lower threshold for hysteresis (auto-calculated if None)
        - high_threshold: Higher threshold for hysteresis (auto-calculated if None)

        Returns:
        - Binary edge map
        """

        def gaussian_kernel(sigma):
            """Create Gaussian kernel based on sigma"""
            size = int(6 * sigma + 1)
            if size % 2 == 0:
                size += 1
            x = np.arange(-(size // 2), size // 2 + 1)
            x, y = np.meshgrid(x, x)
            g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            return g / g.sum()

        def sobel_filters():
            """Create Sobel kernels for x and y directions"""
            Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
            Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
            return Kx, Ky

        def non_maximum_suppression(mag, ang):
            """
            Perform non-maximum suppression on gradient magnitude
            using gradient direction
            """
            height, width = mag.shape
            result = np.zeros_like(mag)

            # Quantize angles to 4 directions (0, 45, 90, 135 degrees)
            ang = np.rad2deg(ang) % 180
            ang = np.floor((ang + 22.5) / 45) * 45

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    if ang[i, j] == 0:
                        neighbors = [mag[i, j - 1], mag[i, j + 1]]
                    elif ang[i, j] == 45:
                        neighbors = [mag[i - 1, j + 1], mag[i + 1, j - 1]]
                    elif ang[i, j] == 90:
                        neighbors = [mag[i - 1, j], mag[i + 1, j]]
                    else:  # 135 degrees
                        neighbors = [mag[i - 1, j - 1], mag[i + 1, j + 1]]

                    if mag[i, j] >= max(neighbors):
                        result[i, j] = mag[i, j]

            return result

        def hysteresis_thresholding(img, low, high):
            """
            Apply hysteresis thresholding to connect edges
            """
            height, width = img.shape

            strong = img >= high
            weak = (img >= low) & (img < high)

            result = np.zeros_like(img)
            result[strong] = 1

            dx = [-1, -1, -1, 0, 0, 1, 1, 1]
            dy = [-1, 0, 1, -1, 1, -1, 0, 1]

            while True:
                changed = False
                for i in range(1, height - 1):
                    for j in range(1, width - 1):
                        if weak[i, j] and not result[i, j]:
                            for k in range(8):
                                if result[i + dx[k], j + dy[k]]:
                                    result[i, j] = 1
                                    changed = True
                                    break
                if not changed:
                    break

            return result

        kernel = gaussian_kernel(sigma)
        smoothed = convolve(image.astype(float), kernel)

        Kx, Ky = sobel_filters()
        Ix = convolve(smoothed, Kx)
        Iy = convolve(smoothed, Ky)

        magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
        angle = np.arctan2(Iy, Ix)

        suppressed = non_maximum_suppression(magnitude, angle)

        if low_threshold is None:
            low_threshold = np.percentile(suppressed, 10)
        if high_threshold is None:
            high_threshold = np.percentile(suppressed, 30)

        edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)

        return (edges * 255).astype(np.uint8)

    @staticmethod
    def custom_dbscan(points, eps, min_samples):
        """
        DBSCAN implementation for ball candidates clustering
        """
        n_points = len(points)
        labels = np.full(n_points, -1)
        cluster_id = 0

        def find_neighbors(point_idx):
            distances = np.sqrt(np.sum((points - points[point_idx]) ** 2, axis=1))
            return np.where(distances <= eps)[0]

        for i in range(n_points):
            if labels[i] != -1:
                continue

            neighbors = find_neighbors(i)
            if len(neighbors) < min_samples:
                continue

            cluster_id += 1
            labels[i] = cluster_id

            neighbors = list(neighbors)
            for j in range(len(neighbors)):
                point_idx = neighbors[j]
                if labels[point_idx] == -1:
                    labels[point_idx] = cluster_id
                    new_neighbors = find_neighbors(point_idx)
                    if len(new_neighbors) >= min_samples:
                        neighbors.extend([n for n in new_neighbors if n not in neighbors])

        return labels

    @staticmethod
    def detect_balls(image_path, max_display_size=800):
        """
        ball detection using multiple preprocessing techniques and adaptive thresholding
        """
        # read and resize image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError("Could not read the image")

        original_height, original_width = original.shape[:2]
        scale = 1.0
        if max(original_height, original_width) > max_display_size:
            scale = max_display_size / max(original_height, original_width)
            working_image = cv2.resize(original, (int(original_width * scale),
                                                  int(original_height * scale)))
        else:
            working_image = original.copy()

        # Convert to different color spaces
        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)

        # Create multiple preprocessing versions
        preprocessed_images = []

        # 1. Standard grayscale with contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        preprocessed_images.append(enhanced_gray)

        # 2. Color-based detection for specific ball colors
        # Tennis ball (yellow/green)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        preprocessed_images.append(yellow_mask)

        # 3. Difference of Gaussians
        g1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
        g2 = cv2.GaussianBlur(gray, (9, 9), 2.0)
        dog = cv2.subtract(g1, g2)
        preprocessed_images.append(dog)

        # 4. Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(adaptive_thresh)

        # Collect ball candidates
        candidates = []

        # Multiple threshold ranges for edge detection
        threshold_pairs = [
            (0.5, 1.5),  # More sensitive
            (0.33, 1.33),  # Standard
            (0.25, 1.25)  # Very sensitive
        ]

        for preprocessed in preprocessed_images:
            denoised = cv2.bilateralFilter(preprocessed, 9, 75, 75)

            edges_combined = None
            for lower_mult, upper_mult in threshold_pairs:
                for kernel_size in [(5, 5), (7, 7), (9, 9)]:
                    blurred = cv2.GaussianBlur(denoised, kernel_size, 0)
                    median = np.median(blurred)
                    lower = int(max(0, lower_mult * median))
                    upper = int(min(255, upper_mult * median))

                    edges = BallShooting.manual_canny(blurred,
                                         sigma=1.0,
                                         low_threshold=lower,
                                         high_threshold=upper)

                    if edges_combined is None:
                        edges_combined = edges
                    else:
                        edges_combined = cv2.bitwise_or(edges_combined, edges)

            contours, _ = cv2.findContours(edges_combined,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Lower minimum area threshold
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)

                if hull_area == 0:
                    continue

                convexity = area / hull_area
                confidence = (circularity + convexity) / 2

                # More lenient thresholds
                if circularity > 0.6 and convexity > 0.7:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    candidates.append([x, y, radius, confidence])

        # If no candidates found, return original image
        if not candidates:
            return original, []

        candidates = np.array(candidates)

        # More lenient clustering
        eps = np.mean(candidates[:, 2]) * 1.2
        labels = BallShooting.custom_dbscan(candidates[:, :2], eps=eps, min_samples=1)

        # Process clusters
        balls = []
        display_image = original.copy()
        valid_centers = []

        for label in set(labels):
            if label == -1:
                continue

            cluster_mask = labels == label
            cluster_candidates = candidates[cluster_mask]

            best_idx = np.argmax(cluster_candidates[:, 3])
            x, y, radius, conf = cluster_candidates[best_idx]

            # Additional validation checks
            if conf < 0.7:  # Minimum confidence threshold
                continue

            if radius < 5:  # Minimum radius check
                continue

            if scale != 1.0:
                x = x / scale
                y = y / scale
                radius = radius / scale

                # Check for overlapping detections
            is_duplicate = False
            for existing_x, existing_y, existing_r, _ in balls:
                distance = np.sqrt((existing_x - x) ** 2 + (existing_y - y) ** 2)
                if distance < max(radius, existing_r):
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            center = (int(x), int(y))
            radius = int(radius)
            balls.append((center[0], center[1], radius, conf))
            valid_centers.append(center)

            cv2.circle(display_image, center, radius, (0, 255, 0), 2)
            cv2.circle(display_image, center, 2, (0, 0, 255), -1)

        return display_image, balls

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

    # def rk4_step(self, state, dt, k_m_ratio):
    #     """
    #     perform one RK4 integration step
    #     """
    #     k1 = self.equations_of_motion(0, state, k_m_ratio)
    #     k2 = self.equations_of_motion(0, state + dt * k1 / 2, k_m_ratio)
    #     k3 = self.equations_of_motion(0, state + dt * k2 / 2, k_m_ratio)
    #     k4 = self.equations_of_motion(0, state + dt * k3, k_m_ratio)
    #     return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def implicit_trapezoidal_step(self, state, dt, k_m_ratio):
        """A-stable implicit trapezoidal method"""

        def residual(next_state):
            f_current = self.equations_of_motion(0, state, k_m_ratio)
            f_next = self.equations_of_motion(0, next_state, k_m_ratio)
            return next_state - state - 0.5 * dt * (f_current + f_next)

        # Initial guess using explicit Euler
        f_current = self.equations_of_motion(0, state, k_m_ratio)
        initial_guess = state + dt * f_current

        # Solve implicit equation
        solution = fsolve(residual, initial_guess)
        return solution

    def adaptive_step(self, state, dt, k_m_ratio, error_tolerance=1e-6):
        """adaptive step size control"""
        # compute solution with step size dt
        solution_h = self.implicit_trapezoidal_step(state, dt, k_m_ratio)

        # compute solution with two steps of size dt/2
        half_step = self.implicit_trapezoidal_step(state, dt / 2, k_m_ratio)
        solution_h2 = self.implicit_trapezoidal_step(half_step, dt / 2, k_m_ratio)

        # estimate error
        error = np.max(np.abs(solution_h - solution_h2))

        # update step size based on error
        if error > error_tolerance:
            # reduce step size
            dt_new = dt * 0.5
            return self.adaptive_step(state, dt_new, k_m_ratio, error_tolerance)
        elif error < error_tolerance / 10:
            # increase step size
            dt_new = min(dt * 2, 0.1)  # cap maximum step size
            return solution_h, dt_new

        return solution_h, dt

    def check_ball_hit(self, x, y, target_x, target_y, target_radius):
        """check if trajectory hits the target ball"""
        distance = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        return distance <= target_radius + self.tolerance

    def calculate_trajectory_hash(self, x0, y0, vx0, vy0, target_x, target_y):
        """hash function for trajectory caching"""
        return hash((round(x0, 2), round(y0, 2),
                    round(vx0, 2), round(vy0, 2),
                    round(target_x, 2), round(target_y, 2)))

    def calculate_trajectory(self, x0, y0, v0, angle, target_x, target_y, target_radius):
        """calculate trajectory with caching"""
        vx0 = v0 * np.cos(angle)
        vy0 = -v0 * np.sin(angle)

        # Debug prints
        print(f"\nCalculating trajectory:")
        print(f"Initial position: ({x0:.2f}, {y0:.2f})")
        print(f"Initial velocity: ({vx0:.2f}, {vy0:.2f})")
        print(f"Target position: ({target_x:.2f}, {target_y:.2f})")
        print(f"Angle: {np.degrees(angle):.2f} degrees")
        print(f"Initial speed: {v0:.2f}")

        # check cache
        traj_hash = self.calculate_trajectory_hash(x0, y0, vx0, vy0, target_x, target_y)
        if traj_hash in self.trajectory_cache:
            print("Using cached trajectory")
            return self.trajectory_cache[traj_hash]

        state = np.array([x0, y0, vx0, vy0])
        trajectory = self.integrate_trajectory(state, target_x, target_y, target_radius)

        if trajectory is not None:
            print(f"Trajectory calculated with {len(trajectory)} points")
        else:
            print("Failed to calculate trajectory")

        # cache result
        self.trajectory_cache[traj_hash] = trajectory
        return trajectory

    def integrate_trajectory(self, initial_state, target_x, target_y, target_radius):
        """Integration with adaptive step size"""
        max_steps = 10000
        trajectory = np.zeros((max_steps, 2))
        state = initial_state.copy()
        dt = self.dt
        i = 0

        try:
            while i < max_steps:
                trajectory[i] = state[:2]

                # Check for target hit or out of bounds
                if self.check_ball_hit(state[0], state[1], target_x, target_y, target_radius):
                    return trajectory[:i + 1]

                if self.is_out_of_bounds(state, target_x, target_y):
                    return trajectory[:i + 1]

                # Perform adaptive step
                state, dt = self.adaptive_step(state, dt, self.k_m_ratio)
                i += 1

            return trajectory[:i]

        except Exception as e:
            print(f"Error in trajectory integration: {e}")
            return None

    def shooting_method(self, x0, y0, target_x, target_y, target_radius, v0_initial=200, max_iter=20):
        """
        shooting method using Newton's method with numerical Jacobian
        """
        # Calculate minimum required velocity based on ballistic trajectory
        min_v0_required = np.sqrt(2 * self.g * (y0 - target_y) +
                                  (self.g * (target_x - x0) ** 2) / (2 * (y0 - target_y)))

        velocities_to_try = [
            max(v0_initial, min_v0_required),
            max(v0_initial * 1.2, min_v0_required),
            max(v0_initial * 1.5, min_v0_required)
        ]

        angles_to_try = [
            np.arctan2(-(target_y - y0), target_x - x0),  # direct angle
            np.arctan2(-(target_y - y0), target_x - x0) - 0.2,  # slightly lower
            np.arctan2(-(target_y - y0), target_x - x0) + 0.2  # slightly higher
        ]

        best_trajectory = None
        best_distance = float('inf')
        best_params = None

        for v0_try in velocities_to_try:
            for initial_theta in angles_to_try:
                try:
                    theta = initial_theta
                    v0 = v0_try
                    tol = 1e-3
                    min_v0 = v0_try * 0.8
                    max_v0 = v0_try * 1.2
                    min_theta, max_theta = -np.pi / 2.2, np.pi / 2.2  # slightly restricted angle range

                    print(f"\nTrying with velocity {v0:.2f} and angle {np.degrees(theta):.2f}")

                    for iteration in range(max_iter):
                        trajectory = self.calculate_trajectory(x0, y0, v0, theta, target_x, target_y, target_radius)
                        if trajectory is None or len(trajectory) < 2:
                            v0 = min(v0 * 1.1, max_v0)
                            continue

                        final_pos = trajectory[-1]
                        current_distance = np.sqrt((final_pos[0] - target_x) ** 2 + (final_pos[1] - target_y) ** 2)

                        # update best trajectory if this one is closer
                        if current_distance < best_distance:
                            best_distance = current_distance
                            best_trajectory = trajectory
                            best_params = (theta, v0)

                        # check if we hit the target
                        if self.check_ball_hit(final_pos[0], final_pos[1], target_x, target_y, target_radius):
                            return theta, v0, trajectory

                        # rest of Newton's method remains the same
                        F = np.array([final_pos[0] - target_x, final_pos[1] - target_y])
                        J = self.calculate_jacobian(x0, y0, v0, theta, target_x, target_y, target_radius)

                        if J is None:
                            v0 = min(v0 * 1.1, max_v0)
                            continue

                        J = J + np.eye(2) * 1e-4

                        try:
                            delta = np.linalg.solve(J, -F)
                            delta = np.clip(delta, -0.1, 0.1)  # more conservative step size

                            theta_new = np.clip(theta + delta[0], min_theta, max_theta)
                            v0_new = np.clip(v0 + delta[1], min_v0, max_v0)

                            if np.abs(theta_new - theta) < tol and np.abs(v0_new - v0) < tol:
                                if best_distance < target_radius * 2:  # if we're reasonably close
                                    return best_params[0], best_params[1], best_trajectory
                                break  # try next initial conditions

                            theta, v0 = theta_new, v0_new

                        except np.linalg.LinAlgError:
                            v0 = min(v0 * 1.1, max_v0)

                except Exception as e:
                    print(f"Attempt failed: {str(e)}")
                    continue

        # if we have a best trajectory that's reasonably close, return it
        if best_trajectory is not None and best_distance < target_radius * 3:
            return best_params[0], best_params[1], best_trajectory

        raise ValueError("Failed to find a valid trajectory")

    def calculate_jacobian(self, x0, y0, v0, theta, target_x, target_y, target_radius):
        """
        calculate numerical Jacobian for the shooting method
        """
        try:
            # small perturbations
            d_theta = 1e-4
            d_v0 = 1e-2

            # base trajectory
            traj_base = self.calculate_trajectory(x0, y0, v0, theta, target_x, target_y, target_radius)

            # perturbed trajectories
            traj_theta_plus = self.calculate_trajectory(x0, y0, v0, theta + d_theta, target_x, target_y, target_radius)
            traj_theta_minus = self.calculate_trajectory(x0, y0, v0, theta - d_theta, target_x, target_y, target_radius)
            traj_v0_plus = self.calculate_trajectory(x0, y0, v0 + d_v0, theta, target_x, target_y, target_radius)
            traj_v0_minus = self.calculate_trajectory(x0, y0, v0 - d_v0, theta, target_x, target_y, target_radius)

            # check if all trajectories are valid
            if any(traj is None or len(traj) < 2 for traj in
                   [traj_base, traj_theta_plus, traj_theta_minus, traj_v0_plus, traj_v0_minus]):
                return None

            # calculate partial derivatives
            dx_dtheta = (traj_theta_plus[-1][0] - traj_theta_minus[-1][0]) / (2 * d_theta)
            dy_dtheta = (traj_theta_plus[-1][1] - traj_theta_minus[-1][1]) / (2 * d_theta)
            dx_dv0 = (traj_v0_plus[-1][0] - traj_v0_minus[-1][0]) / (2 * d_v0)
            dy_dv0 = (traj_v0_plus[-1][1] - traj_v0_minus[-1][1]) / (2 * d_v0)

            return np.array([
                [dx_dtheta, dx_dv0],
                [dy_dtheta, dy_dv0]
            ])

        except Exception as e:
            print(f"Jacobian calculation failed: {str(e)}")
            return None

    def simulate_game(self, image_path, initial_velocity=200):
        """simulate with error handling and progress tracking"""
        try:
            # detect balls and setup initial parameters
            image, ball_positions = self.detect_balls(image_path)
            if not ball_positions:
                raise ValueError("No balls detected")

            # set initial height for bounds checking
            self.initial_height = image.shape[0]
            shooting_pos = (0, self.initial_height)

            # sort balls by distance from shooting position
            ball_positions.sort(key=lambda pos: np.sqrt(
                (pos[0] - shooting_pos[0]) ** 2 + (pos[1] - shooting_pos[1]) ** 2
            ))

            trajectories = []
            print(f"Found {len(ball_positions)} targets")

            for i, (x, y, radius, _) in enumerate(ball_positions):
                max_retries = 3
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        current_velocity = initial_velocity * (1 + retry_count * 0.5)

                        angle, velocity, traj = self.shooting_method(
                            shooting_pos[0], shooting_pos[1],
                            x, y, radius, current_velocity
                        )

                        if traj is not None and len(traj) > 0:
                            trajectories.append(traj)
                            print(f"Successfully calculated trajectory for target {i + 1}")
                            break

                    except Exception as e:
                        print(f"Attempt {retry_count + 1} failed for target {i + 1}: {e}")
                        retry_count += 1
                        continue

                if retry_count == max_retries:
                    print(f"Failed to hit target {i + 1} after {max_retries} attempts")

            # create and setup the figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            self.setup_plots(ax1, ax2, image, ball_positions, shooting_pos)

            # pre-calculate animation frames
            print("Preparing animation frames...")
            frames = self.prepare_animation_frames(trajectories)

            print("Creating animation...")
            anim = FuncAnimation(
                fig,
                self.animate_frame,
                frames=len(frames),
                interval=20,
                blit=True,
                fargs=(ax2, frames)
            )

            return fig, anim

        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            raise

    def prepare_animation_frames(self, trajectories):
        """pre-calculate animation frames"""
        frames = []
        for traj in trajectories:
            num_points = len(traj)
            for i in range(0, num_points, max(1, num_points // 100)):
                frames.append(traj[:i + 1])
        return frames

    def is_out_of_bounds(self, state, target_x, target_y):
        """check if trajectory point is out of valid simulation bounds with adaptive boundaries"""
        x, y = state[0], state[1]

        if self.initial_height is None:
            raise ValueError("initial_height not set. Please run simulate_game first.")

        # more lenient bounds
        height_factor = 2.0
        width_factor = 3.0

        max_height = self.initial_height * height_factor
        max_width = max(target_x * width_factor, self.initial_height * width_factor)

        # special handling for y-coordinate
        # allow trajectory to go below target only if it's still moving towards the target
        if y < target_y - 50:  # allow some margin below target
            if x < target_x:  # if we haven't reached target x-coordinate yet
                return True

        # basic bounds checking
        if x < -50 or x > max_width + 50 or y > max_height + 50:
            print(f"Out of bounds: x={x:.2f}, y={y:.2f}")
            print(f"Bounds: width={max_width:.2f}, height={max_height:.2f}")
            return True

        return False

    def calculate_trajectory_error(self, trajectory, target_x, target_y):
        """calculate minimum distance between trajectory and target"""
        dx = trajectory[:, 0] - target_x
        dy = trajectory[:, 1] - target_y
        distances = np.sqrt(dx ** 2 + dy ** 2)
        return np.min(distances)

    def setup_plots(self, ax1, ax2, image, ball_positions, shooting_pos, initial_velocity=200):
        """setup both plot axes for animation with enhanced information display"""
        # detection plot
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Detected Balls")
        ax1.axis('off')

        # simulation plot
        ax2.set_xlim(0, image.shape[1])
        ax2.set_ylim(image.shape[0], 0)  # invert y-axis
        ax2.set_title("Ball Shooting Simulation")
        ax2.set_xlabel("X Position (pixels)")
        ax2.set_ylabel("Y Position (pixels)")
        ax2.grid(True, alpha=0.3)

        # add shooting position and targets
        marker_size = min(image.shape) / 50
        ax2.add_patch(patches.Circle(shooting_pos, marker_size, color='green', label='Shooting Position'))

        # add targets with distance information
        for i, (x, y, radius, *_) in enumerate(ball_positions):
            ax2.add_patch(patches.Circle((x, y), radius, color='red', alpha=0.6))
            # calculate distance from shooter to target
            distance = np.sqrt((x - shooting_pos[0]) ** 2 + (y - shooting_pos[1]) ** 2)
            ax2.text(x + radius, y - radius, f'Target {i + 1}\nDist: {distance:.1f}px',
                     fontsize=8, ha='left', va='bottom')

        # add trajectory line and moving ball
        self.trajectory_line, = ax2.plot([], [], 'b-', label='Trajectory')
        self.moving_ball = ax2.add_patch(
            patches.Circle(shooting_pos, marker_size, color='blue', label='Ball')
        )

        # create text boxes for dynamic information
        self.info_text = ax2.text(
            0.02, 0.98, '',
            transform=ax2.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            fontsize=8,
            verticalalignment='top'
        )

        # add simulation parameters box
        params_text = (
            f"Simulation Parameters:\n"
            f"Initial Velocity: {initial_velocity} px/s\n"
            f"Gravity: {self.g:.1f} px/s²\n"
            f"Air Resistance (k/m): {self.k_m_ratio:.4f}\n"
            f"Time Step: {self.dt:.3f} s"
        )
        ax2.text(
            0.98, 0.98, params_text,
            transform=ax2.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='right'
        )

        ax2.legend(loc='lower right')

    def animate_frame(self, frame_idx, ax, frames):
        """animate a single frame with enhanced information"""
        trajectory = frames[frame_idx]
        self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])

        if len(trajectory) > 0:
            current_pos = trajectory[-1]
            self.moving_ball.center = (current_pos[0], current_pos[1])

            # calculate current velocity (if we have at least 2 points)
            if len(trajectory) > 1:
                dt = self.dt
                prev_pos = trajectory[-2]
                vx = (current_pos[0] - prev_pos[0]) / dt
                vy = (current_pos[1] - prev_pos[1]) / dt
                v_magnitude = np.sqrt(vx ** 2 + vy ** 2)
                angle = np.arctan2(-vy, vx)  # negative vy because y-axis is inverted

                # update info text
                info_str = (
                    f"Ball Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f})\n"
                    f"Velocity: {v_magnitude:.1f} px/s\n"
                    f"Angle: {np.degrees(angle):.1f}°"
                )
                self.info_text.set_text(info_str)

        return self.trajectory_line, self.moving_ball, self.info_text


if __name__ == "__main__":
    game = BallShooting()
    try:
        print("Starting simulation...")
        fig, anim = game.simulate_game('media/canva.png')
        print("Animation created, showing plot...")
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()