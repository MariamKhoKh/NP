import numpy as np
import cv2
from scipy.optimize import least_squares


class BallTrajectoryAnalyzer:
    def __init__(self):
        self.positions = []
        self.timestamps = []
        self.background = None
        self.g = 9.81
        self.pixel_to_meter = 0.01  # adjust 'pixel_to_meter' if you know real length from point A to point B
        self.drag_coefficient = None
        self.mass = None
        self.avg_velocity = None

    def create_background_model(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError("Error: Cannot open video file")

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
        return False

    def detect_ball(self, frame):

        """   detects the ball in a single video frame by comparing it with the background model.
              args:
              - frame: single frame of the video.

              returns:
              - (x, y) coordinates of the ball's center if detected, none otherwise.
        """

        if self.background is None:
            return None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_background = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_frame, gray_background)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_match = None
        best_circularity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # Min area threshold
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

    def analyze_video(self, video_path, output_path=None):
        """  analyzes the video to detect the ball, calculate trajectory, and annotate results.
               args:
               - video_path (input)
               - output_path

               returns:
               - True if analysis was successful, False otherwise.
        """

        if not self.create_background_model(video_path):
            print("Failed to create background model")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open video")
            return False

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        self.positions = []
        self.timestamps = []

        """Pass 1: Detect ball positions"""
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ball_pos = self.detect_ball(frame)

            if ball_pos is not None:
                self.positions.append(ball_pos)
                self.timestamps.append(frame_count / fps)

            frame_count += 1

        cap.release()

        """ Pass 2: Annotate video with result parameters"""
        if len(self.positions) > 2:
            self.calculate_drag_and_mass()
            velocities = self.calculate_velocities()

            cap = cv2.VideoCapture(video_path)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count < len(self.positions):
                    ball_pos = self.positions[frame_count]
                    cv2.circle(frame, ball_pos, 5, (0, 255, 0), -1)


                if len(self.positions) > 1:
                    for i in range(1, len(self.positions)):
                        pt1 = self.positions[i - 1]
                        pt2 = self.positions[i]
                        cv2.line(frame, pt1, pt2, (255, 0, 0), 2)


                if self.drag_coefficient is not None and self.mass is not None:
                    text_1 = f"Drag Coefficient (k): {self.drag_coefficient:.4e}"
                    text_2 = f"Mass (m): {self.mass:.4f} kg"
                    text_3 = f"Avg Velocity: {self.avg_velocity:.4f} m/s"

                    cv2.putText(frame, text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, text_3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if output_path:
                    out.write(frame)

                cv2.imshow('Ball Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()

        return True

    def calculate_drag_and_mass(self):
        """estimates the drag coefficient and mass of the ball using the recorded positions and timestamps.
           solves equations of motion under the influence of drag and gravity."""
        positions = np.array(self.positions)
        timestamps = np.array(self.timestamps)

        def equations(params):
            k, m = params
            v_x = np.gradient(positions[:, 0] * self.pixel_to_meter, timestamps)
            v_y = np.gradient(positions[:, 1] * self.pixel_to_meter, timestamps)
            dv_x_dt = np.gradient(v_x, timestamps)
            dv_y_dt = np.gradient(v_y, timestamps)

            drag_x = -k * v_x / m
            drag_y = -k * v_y / m - self.g

            return np.hstack([dv_x_dt - drag_x, dv_y_dt - drag_y])

        initial_guess = [0.1, 1.0]
        result = least_squares(equations, initial_guess, bounds=(0, np.inf))
        k, m = result.x

        self.drag_coefficient = k
        self.mass = m
        return k, m

    def calculate_velocities(self):
        """ calculates the velocities of the ball at each timestamp
            and computes the average velocity."""
        velocities = []
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            dt = self.timestamps[i] - self.timestamps[i - 1]
            v = np.sqrt((dx * self.pixel_to_meter / dt) ** 2 + (dy * self.pixel_to_meter / dt) ** 2)
            velocities.append(v)

        self.avg_velocity = np.mean(velocities)
        return velocities


if __name__ == "__main__":
    video_path = "canva.mp4"
    output_path = "output_video.avi"
    analyzer = BallTrajectoryAnalyzer()

    if analyzer.analyze_video(video_path, output_path):
        print("Ball tracking and parameter estimation completed.")
    else:
        print("failed to process the video.")
