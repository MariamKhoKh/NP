import numpy as np
import cv2


def colorConvert(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Capture the video
cap = cv2.VideoCapture('Media/vtest.avi')
if not cap.isOpened():
    raise IOError("Error: Unable to open the video file.")

# Randomly select 30 unique frame positions
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = np.unique((frame_count * np.random.uniform(size=30)).astype(int))

# Capture and store selected frames
frames = []
for i in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

cap.release()

# Check if frames were captured
if len(frames) == 0:
    raise ValueError("No frames were captured. Check the video file.")

# Calculate median frame
frame_median = np.median(frames, axis=0).astype(np.uint8)
gray_frame_median = cv2.cvtColor(frame_median, cv2.COLOR_BGR2GRAY)

# Define the output video writer
frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('Detect.avi', fourcc, 20.0, (frame_width, frame_height))

# Reopen the video for processing
cap = cv2.VideoCapture('Media/vtest.avi')

# Initialize object tracking variables
object_id_count = 0
tracked_objects = {}  # Dictionary to store object IDs and their centroids
object_disappeared = {}  # Track "disappeared" status for each object

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Absolute difference between current frame and median frame
    dframe = cv2.absdiff(gray_frame, gray_frame_median)

    # Apply Gaussian Blur to reduce noise
    blur_frame = cv2.GaussianBlur(dframe, (11, 11), 0)

    # Binarize frame using OTSU's thresholding
    _, threshold_frame = cv2.threshold(blur_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dictionary to store the current frame's object centroids and their IDs
    current_frame_centroids = {}

    # Process each contour
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Filter small contours
            x, y, width, height = cv2.boundingRect(contour)
            cx = int(x + width / 2)
            cy = int(y + height / 2)
            centroid = (cx, cy)

            # Match the current centroid with previously tracked objects
            closest_object_id = None
            min_distance = float("inf")
            for object_id, data in tracked_objects.items():
                prev_centroid = data["centroid"]
                distance = np.linalg.norm(np.array(prev_centroid) - np.array(centroid))
                if distance < min_distance and distance < 50:  # Threshold for matching
                    min_distance = distance
                    closest_object_id = object_id

            # If a match is found, use the existing ID; otherwise, assign a new ID
            if closest_object_id is not None:
                current_frame_centroids[closest_object_id] = centroid
            else:
                current_frame_centroids[object_id_count] = centroid
                tracked_objects[object_id_count] = {"centroid": centroid, "speed": 0}
                object_disappeared[object_id_count] = 0
                object_id_count += 1

            # Speed calculation: check if the object has a previous position
            if closest_object_id is not None:
                prev_centroid = tracked_objects[closest_object_id]["centroid"]
                speed = np.linalg.norm(np.array(prev_centroid) - np.array(centroid))
                tracked_objects[closest_object_id]["speed"] = speed  # Update speed in pixels/frame
                tracked_objects[closest_object_id]["centroid"] = centroid  # Update position
                object_disappeared[closest_object_id] = 0  # Reset disappearance counter

                # Display object ID and speed on the frame
                speed_text = f"Speed: {speed:.2f} px/frame"
                cv2.putText(frame, f"ID: {closest_object_id}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Display new object ID
                cv2.putText(frame, f"ID: {object_id_count - 1}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Speed: 0 px/frame", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (x, y), (x + width, y + height), (123, 0, 255), 2)

    # Update tracked objects with current frame centroids
    for object_id in list(tracked_objects.keys()):  # Use list to avoid runtime errors
        if object_id in current_frame_centroids:
            tracked_objects[object_id]["centroid"] = current_frame_centroids[object_id]
        else:
            object_disappeared[object_id] += 1
            # Remove object if it disappears for a few consecutive frames
            if object_disappeared[object_id] > 5:
                tracked_objects.pop(object_id)
                object_disappeared.pop(object_id)

    # Display the real-time object count on the frame
    cv2.putText(frame, f"Object Count: {len(tracked_objects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame to the output video
    video_writer.write(frame)

cap.release()
video_writer.release()

# Display the processed video with the original speed
# Get FPS from the original video
original_cap = cv2.VideoCapture('Media/vtest.avi')
fps = original_cap.get(cv2.CAP_PROP_FPS)
original_cap.release()

# Play the output video 'Detect.avi' at the original FPS
cap = cv2.VideoCapture('Detect.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Display the frame with the original speed
    cv2.imshow('Processed Video Frame', frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # Display with original speed
        break

cap.release()
cv2.destroyAllWindows()

print("Processing complete. The output video is saved as 'Detect.avi'.")
