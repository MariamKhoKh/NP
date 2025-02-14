import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches


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
    def detect_balls(image_path, max_display_size=800):
        """
        Detect balls using multiple preprocessing techniques to improve detection.
        """
        # read image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError("Could not read the image")

        # store original size
        original_height, original_width = original.shape[:2]

        # calculate scaling
        scale = 1.0
        if max(original_height, original_width) > max_display_size:
            scale = max_display_size / max(original_height, original_width)
            working_image = cv2.resize(original, (int(original_width * scale), int(original_height * scale)))
        else:
            working_image = original.copy()

        # convert to different color spaces and create grayscale versions
        gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)

        # create multiple preprocessing versions
        preprocessed_images = []

        # 1. standard grayscale with contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        preprocessed_images.append(enhanced_gray)

        # 2. adaptive thresholding version
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(adaptive_thresh)

        # 3. value channel from HSV with enhancement
        value_channel = hsv[:, :, 2]
        enhanced_value = clahe.apply(value_channel)
        preprocessed_images.append(enhanced_value)

        # process each version
        all_contours = []
        for preprocessed in preprocessed_images:
            # apply bilateral filter
            denoised = cv2.bilateralFilter(preprocessed, 9, 75, 75)

            # multiple blur scales
            blur_scales = [(7, 7, 1.5), (9, 9, 2), (11, 11, 2.5)]
            edges_combined = None

            for kernel_size_x, kernel_size_y, sigma in blur_scales:
                blurred = cv2.GaussianBlur(denoised, (kernel_size_x, kernel_size_y), sigma)

                # dynamic thresholding for Canny
                median = np.median(blurred)
                lower = int(max(0, (1.0 - 0.33) * median))
                upper = int(min(255, (1.0 + 0.33) * median))

                edges = cv2.Canny(blurred, lower, upper)

                if edges_combined is None:
                    edges_combined = edges
                else:
                    edges_combined = cv2.bitwise_or(edges_combined, edges)

            # enhance edges
            kernel = np.ones((3, 3), np.uint8)
            edges_combined = cv2.dilate(edges_combined, kernel, iterations=1)

            # find contours for this version
            contours, _ = cv2.findContours(edges_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)

        # parameters for ball detection
        min_radius = 1
        max_radius = min(working_image.shape[0], working_image.shape[1]) // 2
        # relaxed threshold
        min_circularity = 0.65
        min_convexity = 0.75

        balls = []
        display_image = original.copy()

        for contour in all_contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            # shape metrics calculation
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            convexity = area / hull_area
            confidence_score = (circularity + convexity) / 2

            if circularity > min_circularity and convexity > min_convexity:
                (x, y), radius = cv2.minEnclosingCircle(contour)

                if min_radius <= radius <= max_radius:
                    # scale coordinates if needed
                    if scale != 1.0:
                        x = x / scale
                        y = y / scale
                        radius = radius / scale

                    # check for overlap
                    is_unique = True
                    for (prev_x, prev_y, prev_r, _) in balls:
                        dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                        if dist < max(radius, prev_r):
                            if confidence_score > 0.85:
                                balls.remove((prev_x, prev_y, prev_r, _))
                            else:
                                is_unique = False
                            break

                    if is_unique:
                        center = (int(x), int(y))
                        radius = int(radius)
                        balls.append((int(x), int(y), radius, confidence_score))

                        # Draw detection
                        cv2.circle(display_image, center, radius, (0, 255, 0), 2)
                        cv2.circle(display_image, center, 2, (0, 0, 255), -1)

                        # Add radius measurement
                        text = f"R={radius}px"
                        cv2.putText(display_image, text,
                                    (center[0] - 20, center[1] - radius - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

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
                print(f"Calculating trajectory for target {i + 1}/{len(ball_positions)}")
                try:
                    angle, traj = self.shooting_method(
                        shooting_pos[0], shooting_pos[1],
                        x, y, radius, initial_velocity
                    )
                    if traj is not None and len(traj) > 0:
                        trajectories.append(traj)
                        print(f"Successfully calculated trajectory for target {i + 1}")
                    else:
                        print(f"Warning: Invalid trajectory for target {i + 1}")
                except ValueError as e:
                    print(f"Warning: Failed to hit target {i + 1}: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error for target {i + 1}: {e}")
                    continue

            if not trajectories:
                raise ValueError("No valid trajectories found")

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

    def create_animation(self, image, ball_positions, trajectories, shooting_pos):
        """create animation with performance optimization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Setup plots
        self.setup_plots(ax1, ax2, image, ball_positions, shooting_pos)

        # Pre-calculate animation frames for smoother playback
        frames = self.prepare_animation_frames(trajectories)

        anim = FuncAnimation(
            fig,
            self.animate_frame,
            frames=len(frames),
            interval=20,
            blit=True,
            fargs=(ax2, frames)
        )

        return fig, anim

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

    def setup_plots(self, ax1, ax2, image, ball_positions, shooting_pos):
        """setup both plot axes for animation"""
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
        ax2.grid(True)

        # add shooting position and targets
        marker_size = min(image.shape) / 50
        ax2.add_patch(patches.Circle(shooting_pos, marker_size, color='green', label='Shooting Position'))
        for x, y, radius, _ in ball_positions:
            ax2.add_patch(patches.Circle((x, y), radius, color='red'))

        # add trajectory line and moving ball
        self.trajectory_line, = ax2.plot([], [], 'b-', label='Trajectory')
        self.moving_ball = ax2.add_patch(
            patches.Circle(shooting_pos, marker_size, color='blue', label='Ball'))
        ax2.legend()

    def animate_frame(self, frame_idx, ax, frames):
        """animate a single frame"""
        trajectory = frames[frame_idx]
        self.trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])
        if len(trajectory) > 0:
            self.moving_ball.center = (trajectory[-1, 0], trajectory[-1, 1])
        return self.trajectory_line, self.moving_ball


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
