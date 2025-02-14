import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from scipy.ndimage import convolve


class BallShooting:
    def __init__(self):
        self.g = 9.81
        self.dt = 0.01
        self.tolerance = 5.0
        self.k_m_ratio = 0.001
        self.trajectory_cache = {}
        self.initial_height = None
        self.trajectory_line = None
        self.moving_ball = None
        self.target_patches = []
        self.active_targets = []
        self.target_map = {}  # Map to track target patches
        self.current_trajectory_index = 0
        self.animation = None

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
        Efficient DBSCAN implementation for ball candidates clustering
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
        Enhanced ball detection using multiple preprocessing techniques
        and adaptive thresholding
        """
        # Read and resize image
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
        # Tennis ball (yellow)
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
        Equations of motion with gravity and air resistance
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

    def rk4_step(self, state, dt, k_m_ratio):
        """
        perform one RK4 integration step
        """
        k1 = self.equations_of_motion(0, state, k_m_ratio)
        k2 = self.equations_of_motion(0, state + dt * k1 / 2, k_m_ratio)
        k3 = self.equations_of_motion(0, state + dt * k2 / 2, k_m_ratio)
        k4 = self.equations_of_motion(0, state + dt * k3, k_m_ratio)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

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

        # check cache
        traj_hash = self.calculate_trajectory_hash(x0, y0, vx0, vy0, target_x, target_y)
        if traj_hash in self.trajectory_cache:
            return self.trajectory_cache[traj_hash]

        state = np.array([x0, y0, vx0, vy0])
        trajectory = self.integrate_trajectory(state, target_x, target_y, target_radius)

        # cache result
        self.trajectory_cache[traj_hash] = trajectory
        return trajectory

    def integrate_trajectory(self, initial_state, target_x, target_y, target_radius):
        """RK4 integration with early stopping and error handling"""
        try:
            max_steps = 5000
            trajectory = np.zeros((max_steps, 2))
            state = initial_state.copy()

            for i in range(max_steps):
                trajectory[i] = state[:2]

                # check for target hit
                if self.check_ball_hit(state[0], state[1], target_x, target_y, target_radius):
                    return trajectory[:i + 1]

                # check bounds
                try:
                    if self.is_out_of_bounds(state, target_x, target_y):
                        return trajectory[:i + 1]
                except ValueError as e:
                    print(f"Warning: {e}")
                    return trajectory[:i + 1]

                # perform integration step
                state = self.rk4_step(state, self.dt, self.k_m_ratio)

            return trajectory[:i + 1]

        except Exception as e:
            print(f"Error in trajectory integration: {e}")
            return None

    def shooting_method(self, x0, y0, target_x, target_y, target_radius, v0_initial=100):
        """
        find angle
        velocity adjustment if the current speed isn't enough
        RK4 integration to calculate the actual path with physics (gravity + air resistance)
        """
        # calculate initial distance to target
        dx = target_x - x0
        dy = target_y - y0
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # initial velocity estimation
        v0 = max(v0_initial, np.sqrt(2 * self.g * distance))
        base_angle = np.arctan2(-dy, dx)

        # velocity adjustment loop
        for v_iter in range(15):
            angle, trajectory = self.angle_search(x0, y0, v0, base_angle, target_x, target_y, target_radius)

            if trajectory is None:
                v0 *= 1.2   # increase velocity by 20% and try again
                continue

            final_pos = trajectory[-1]
            if self.check_ball_hit(final_pos[0], final_pos[1], target_x, target_y, target_radius):
                return angle, trajectory

            v0 *= 1.2  # if we didn't hit, increase velocity

        raise ValueError(f"Failed to find solution for target at ({target_x}, {target_y})")

    def angle_search(self, x0, y0, v0, base_angle, target_x, target_y, target_radius):
        """binary search for optimal angle"""
        angle_min = base_angle - np.pi / 3
        angle_max = base_angle + np.pi / 3
        best_angle = None
        best_trajectory = None
        best_error = float('inf')

        for _ in range(20):
            angle = (angle_min + angle_max) / 2
            trajectory = self.calculate_trajectory(x0, y0, v0, angle, target_x, target_y, target_radius)

            if len(trajectory) < 2:
                continue

            error = self.calculate_trajectory_error(trajectory, target_x, target_y)

            if error < best_error:
                best_error = error
                best_angle = angle
                best_trajectory = trajectory

            if error < self.tolerance:
                return angle, trajectory

            if trajectory[-1][1] > target_y:
                angle_min = angle
            else:
                angle_max = angle

        return best_angle, best_trajectory

    def simulate_game(self, image_path, initial_velocity=200):
        """Simulate game with fixed animation settings"""
        try:
            image, ball_positions = self.detect_balls(image_path)
            if not ball_positions:
                raise ValueError("No balls detected")

            self.initial_height = image.shape[0]
            shooting_pos = (0, self.initial_height)

            # Sort balls by distance
            ball_positions.sort(key=lambda pos: np.sqrt(
                (pos[0] - shooting_pos[0]) ** 2 + (pos[1] - shooting_pos[1]) ** 2
            ))

            trajectories = []
            print(f"Found {len(ball_positions)} targets")

            for i, (x, y, radius, _) in enumerate(ball_positions):
                try:
                    angle, traj = self.shooting_method(
                        shooting_pos[0], shooting_pos[1],
                        x, y, radius, initial_velocity
                    )
                    if traj is not None and len(traj) > 0:
                        trajectories.append((traj, (x, y, radius, i)))
                        print(f"Successfully calculated trajectory for target {i + 1}")
                except Exception as e:
                    print(f"Warning: Failed to calculate for target {i + 1}: {e}")
                    continue

            if not trajectories:
                raise ValueError("No valid trajectories found")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            self.setup_plots(ax1, ax2, image, ball_positions, shooting_pos)

            # Initialize target tracking
            self.active_targets = [(x, y, radius, i) for i, (x, y, radius, _) in enumerate(ball_positions)]
            for i, target in enumerate(self.active_targets):
                self.target_map[i] = target

            # Calculate total frames for save_count
            total_frames = sum(len(traj[0]) // 3 + 5 for traj, _ in trajectories)  # Account for step size and pauses

            self.animation = FuncAnimation(
                fig,
                self.animate_frame,
                init_func=self.init_animation,
                frames=self.frame_generator(trajectories),
                interval=10,
                blit=True,
                repeat=False,
                save_count=total_frames,  # Explicitly set save_count
                cache_frame_data=False  # Disable frame caching
            )

            return fig, self.animation

        except Exception as e:
            print(f"Error in simulation: {str(e)}")
            raise

    def init_animation(self):
        """Initialize animation"""
        self.trajectory_line.set_data([], [])
        return (self.trajectory_line, self.moving_ball, self.info_text,
                *self.target_patches)

    def frame_generator(self, trajectories):
        """Generate frames for animation with improved target handling and performance"""
        shooting_pos = (0, self.initial_height)
        step_size = 4  # Increase step size to show fewer frames
        min_points = 10  # Minimum number of points to show per trajectory

        for trajectory, target_info in trajectories:
            x, y, radius, target_idx = target_info
            trajectory_length = len(trajectory)

            # Calculate adaptive step size based on trajectory length
            actual_step = max(1, min(step_size, trajectory_length // min_points))
            points_shown = 0

            while points_shown < trajectory_length:
                # Take larger steps through the trajectory points
                next_point = min(points_shown + actual_step, trajectory_length)
                current_trajectory = trajectory[:next_point]

                if len(current_trajectory) > 0:
                    current_pos = current_trajectory[-1]
                    # Check if target is hit
                    if self.check_ball_hit(current_pos[0], current_pos[1], x, y, radius):
                        yield (current_trajectory, target_idx, True)  # Target hit
                        break

                yield (current_trajectory, target_idx, False)  # Target not hit yet
                points_shown = next_point

            # Reset ball position for next trajectory
            self.moving_ball.center = shooting_pos

            # Reduced pause between trajectories
            for _ in range(5):  # Reduced from 10 to 5 frames of pause
                yield (current_trajectory, target_idx, False)

    def animate_frame(self, frame_data):
        """Optimized frame animation with proper ball updates"""
        if frame_data is None:
            return (self.trajectory_line, self.moving_ball, self.info_text,
                    *self.target_patches)

        trajectory, target_idx, is_hit = frame_data

        # Update only if there's new data
        if len(trajectory) > 0:
            # Update trajectory line
            self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])

            # Update ball position using set_center
            current_pos = trajectory[-1]
            self.moving_ball.center = (current_pos[0], current_pos[1])

            # Calculate velocity only if needed (every 3rd frame)
            if len(trajectory) > 1 and len(trajectory) % 3 == 0:
                prev_pos = trajectory[-2]
                vx = (current_pos[0] - prev_pos[0]) / self.dt
                vy = (current_pos[1] - prev_pos[1]) / self.dt
                v_magnitude = np.sqrt(vx ** 2 + vy ** 2)
                angle = np.arctan2(-vy, vx)

                # Update info text
                info_str = (
                    f"Ball Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f})\n"
                    f"Velocity: {v_magnitude:.1f} px/s\n"
                    f"Angle: {np.degrees(angle):.1f}°\n"
                    f"Remaining Targets: {len(self.active_targets)}"
                )
                self.info_text.set_text(info_str)

        # Handle target hit
        if is_hit and target_idx in self.target_map:
            self.target_patches[target_idx].set_visible(False)
            del self.target_map[target_idx]
            self.active_targets = [target for target in self.active_targets if target[3] != target_idx]

        return (self.trajectory_line, self.moving_ball, self.info_text,
                *self.target_patches)

    def prepare_animation_frames(self, trajectories):
        """pre-calculate animation frames"""
        frames = []
        for traj in trajectories:
            num_points = len(traj)
            for i in range(0, num_points, max(1, num_points // 100)):
                frames.append(traj[:i + 1])
        return frames

    def is_out_of_bounds(self, state, target_x, target_y):
        """check if trajectory point is out of valid simulation bounds"""
        x, y = state[0], state[1]

        # ensure initial_height is set
        if self.initial_height is None:
            raise ValueError("initial_height not set. please run simulate_game first.")

        # define bounds relative to target and image size
        max_height = self.initial_height * 1.5
        max_width = max(target_x * 2, self.initial_height * 2)

        # check bounds
        return (x < 0 or x > max_width or  # horizontal
                y < 0 or y > max_height)  # vertical

    def calculate_trajectory_error(self, trajectory, target_x, target_y):
        """calculate minimum distance between trajectory and target"""
        dx = trajectory[:, 0] - target_x
        dy = trajectory[:, 1] - target_y
        distances = np.sqrt(dx ** 2 + dy ** 2)
        return np.min(distances)

    def setup_plots(self, ax1, ax2, image, ball_positions, shooting_pos, initial_velocity=200):
        """Setup plots with fixed ball shape and improved visualization"""
        # Detection plot
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Detected Balls")
        ax1.axis('off')

        # Simulation plot
        ax2.set_xlim(0, image.shape[1])
        ax2.set_ylim(image.shape[0], 0)
        ax2.set_title("Ball Shooting Simulation")
        ax2.set_xlabel("X Position (pixels)")
        ax2.set_ylabel("Y Position (pixels)")
        ax2.grid(True, alpha=0.3)

        # Add shooting position
        marker_size = min(image.shape) / 50
        shooting_marker = patches.Circle(shooting_pos, marker_size, color='green', label='Shooting Position')
        ax2.add_patch(shooting_marker)

        # Add targets with improved tracking
        self.target_patches = []
        for i, (x, y, radius, _) in enumerate(ball_positions):
            target_patch = patches.Circle((x, y), radius, color='red', alpha=0.6)
            ax2.add_patch(target_patch)
            self.target_patches.append(target_patch)
            distance = np.sqrt((x - shooting_pos[0]) ** 2 + (y - shooting_pos[1]) ** 2)
            ax2.text(x + radius, y - radius, f'Target {i + 1}\nDist: {distance:.1f}px',
                     fontsize=8, ha='left', va='bottom')

        # Create trajectory line
        self.trajectory_line, = ax2.plot([], [], 'b-', label='Trajectory', zorder=1)

        # Create moving ball as Circle patch with proper zorder
        self.moving_ball = patches.Circle((shooting_pos[0], shooting_pos[1]),
                                          marker_size / 2,
                                          color='blue',
                                          label='Ball',
                                          zorder=3)  # Increased zorder to appear above trajectory
        ax2.add_patch(self.moving_ball)

        # Create info text with higher zorder
        self.info_text = ax2.text(
            0.02, 0.98, '',
            transform=ax2.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            fontsize=8,
            verticalalignment='top',
            zorder=4  # Ensure text appears above other elements
        )

        # Add simulation parameters
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
            horizontalalignment='right',
            zorder=4
        )

        # Create proper legend with Circle patches
        legend_elements = [
            patches.Circle((0, 0), marker_size, color='green', label='Shooting Position'),
            patches.Circle((0, 0), marker_size / 2, color='blue', label='Ball'),
            plt.Line2D([0], [0], color='blue', label='Trajectory')
        ]
        ax2.legend(handles=legend_elements, loc='lower right')


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
