import cv2
import numpy as np

# Calculate the Bhattacharyya distance between two histograms
def bhattacharyya_distance(hist1, hist2):
    # Compares two histograms using Bhattacharyya distance
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

# Resample particles based on their weights
def resample_particles(particles, weights):
    # Randomly select particle indices based on their weights
    indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
    # Update the particles array with the resampled particles
    particles[:] = particles[indices]
    # Reset weights to be uniform after resampling
    weights.fill(1.0 / len(weights))

# Predict the new state of the particles
def predict_particles(particles, std_dev=20):
    # Add random Gaussian noise to each particle to simulate motion
    particles += np.random.normal(0, std_dev, particles.shape)

# Open the video file
path_video = "WIN_20231204_16_06_05_Pro.mp4"
cap = cv2.VideoCapture(path_video)

# Read the first frame of the video
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

# Get video properties for saving output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_path = "result.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Manually select the region of interest (ROI) for the face
x, y, w, h = cv2.selectROI('Select the face', frame, fromCenter=False)
cv2.destroyWindow('Select the face')

# Initialize particles randomly within the selected ROI
num_particles = 100
particles = np.column_stack((np.random.uniform(x, x + w, num_particles),np.random.uniform(y, y + h, num_particles)))
# Initialize weights uniformly for all particles
weights = np.ones(num_particles) / num_particles

# Calculate the histogram of the selected face ROI as a reference
face_roi = frame[y:y + h, x:x + w]
reference_histogram = cv2.calcHist([face_roi], [0], None, [256], [0, 256])

# Begin the tracking loop
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break
    # Predict the new state of each particle
    predict_particles(particles)

    # Draw red bounding boxes for each particle on the frame
    for particle in particles:
        pt_x, pt_y = int(particle[0]), int(particle[1])
        cv2.rectangle(frame, (pt_x, pt_y), (pt_x + w, pt_y + h), (0, 0, 255), 1)

    # Update particle weights based on similarity to the reference histogram
    for i, (x_pos, y_pos) in enumerate(particles):
        x_pos = int(x_pos)
        y_pos = int(y_pos)
        # Check if the particle is within frame bounds
        if x_pos + w > frame.shape[1] or y_pos + h > frame.shape[0]:
            weights[i] = 0
            continue
        # Calculate the histogram for the current particle's ROI
        particle_roi = frame[y_pos:y_pos + h, x_pos:x_pos + w]
        particle_histogram = cv2.calcHist([particle_roi], [0], None, [256], [0, 256])
        # Update the weight based on Bhattacharyya distance
        weights[i] = 1.0 - bhattacharyya_distance(reference_histogram, particle_histogram)

    # Normalize the weights to sum to 1
    weights += 1.e-500  # Prevent division by zero
    weights /= sum(weights)

    # Resample the particles based on the updated weights
    resample_particles(particles, weights)

    # Estimate the new state as the mean of the particle distribution
    x_estimated = int(np.mean(particles[:, 0]))
    y_estimated = int(np.mean(particles[:, 1]))

    # Draw a green bounding box around the estimated position
    cv2.rectangle(frame, (x_estimated, y_estimated), (x_estimated + w, y_estimated + h), (0, 255, 0), 1)

    # Display the tracking results
    cv2.imshow('Tracking', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Write the frame to the output file
    out.write(frame)
# Release everything when job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
