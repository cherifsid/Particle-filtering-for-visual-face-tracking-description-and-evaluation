# Hand Gesture Tracking with Particle Filtering

## Overview
This Python script implements a particle filter-based hand gesture tracking algorithm using OpenCV. It tracks a face in a video stream by predicting and updating the state of particles representing potential face positions.

## Features
1. **Particle Filter Implementation**
   - Uses a particle filter for dynamic state estimation in a video stream.
   - Handles face tracking through particle representation.

2. **Real-Time Tracking**
   - Performs real-time tracking of a face with bounding box visualization.

3. **User Interaction**
   - Allows manual selection of the initial region of interest (ROI) around the face.

4. **Robust and Adaptable**
   - Capable of adapting to occlusions and varying face appearances.



## Usage
1. Run the script.
2. Select the face ROI in the first frame of the video.
![Screenshot](/results/leroi.png)
3. Press 'Enter' to start tracking.
4. Press 'q' to quit.

## Limitations
- Higher computational cost with increased number of particles.
- Sensitive to initialization and parameter settings.
- Primarily focused on face tracking, limiting its use in broader contexts.

## Example Outputs
![Screenshot](/results/ezgif-5-b9bf1a6753.gif)



