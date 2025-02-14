import cv2
import numpy as np
from scipy.ndimage import convolve


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

                edges = manual_canny(blurred,
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
    eps = np.mean(candidates[:, 2]) * 0.8
    labels = custom_dbscan(candidates[:, :2], eps=eps, min_samples=1)

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


def main():
    import os

    image_path = "media/2.jpg"

    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    try:
        display_image, balls = detect_balls(image_path)

        print(f"Detected {len(balls)} ball(s):")
        for i, (x, y, radius, confidence) in enumerate(balls, start=1):
            print(f"  Ball {i}: Center=({x}, {y}), Radius={radius}, "
                  f"Confidence={confidence:.2f}")

        cv2.imshow("Detected Balls", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
