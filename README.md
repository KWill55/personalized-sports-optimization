

## Table of Contents
- [Overview](#overview)
- [Project Aims](#project-aims)
- [Physical Setup](#physical-setup)
- [Phase 1: Record 3D Free Throw Data](#phase-1-record-3d-free-throw-data)
- [Phase 2: Extract Key Metrics](#phase-2-extract-key-metrics)
- [Phase 3: Analyze Data to Produce Insights](#phase-3-analyze-data-to-produce-insights)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)

# Personalized Sports Optimization

This project is part of the Robotic Vision Lab at Brigham Young University, directed by Dr. DJ Lee.

**Goal:**  
Use computer vision and machine learning to help basketball players improve their free throw consistency by delivering personalized, biomechanically-informed feedback.

Rather than imitating a one-size-fits-all NBA shooting form, we aim to identify the **individual optimal form** for each athlete using 3D motion analysis and AI.


## Project Aims

- Use 3D motion data to extract athlete-specific biomechanics.
- Track basketball trajectories using synchronized multi-camera setup.
- Build models that learn patterns between shooting form and performance.
- Deliver actionable insights tailored to each athlete's unique style.
 

## Physical Setup

**Cameras:**
- **2 Athlete-Tracking Cameras:** Capture full-body motion to extract joint angles and pose.
- **1 Ball-Tracking Camera:** Focuses on ball trajectory and hoop interaction.

**Flash Synchronization System:**
- An Arduino Uno equipped with IR and LEDs signals all cameras simultaneously.
- Used for accurate multi-camera synchronization at the start/end of each shot.


## Phase 1: Record 3D Free Throw Data

**Purpose:**  
Capture synchronized video data of the athlete and basketball to prepare for motion analysis.

**Outputs:**
- Trimmed, synchronized video files (from all 3 angles)
- Time-aligned frames based on flash sync

**Steps:**
- Record athlete motion using two cameras.
- Record ball trajectory using a third camera.
- Use LED flash to synchronize the video timelines.
- Automatically trim videos to isolate free throw sequences.


## Phase 2: Extract Key Metrics

**Purpose:**  
Extract biomechanical and performance metrics for each free throw.

**Outputs:**
- CSV files with athlete joint angles and ball trajectory data.
- Combined dataset: pose + trajectory + outcome.
- Feature matrix `X` and target labels `y` for ML models.

**Metrics:**
- Elbow angle, release angle, shoulder-hip alignment, etc.
- Arc, speed, distance, and angle of ball trajectory.


## Phase 3: Analyze Data to Produce Insights

**Purpose:**  
Use the extracted features to train machine learning models that predict shot outcomes and recommend adjustments.

**Outputs:**
- Classification reports on shot success prediction.
- Visualizations of key trends and biomechanical efficiency.
- Personalized suggestions (e.g., elbow angle optimization).

**Planned Models:**
- Logistic Regression
- Decision Trees / Random Forests
- Gradient Boosting
- SVM / KNN

## Technologies Used

- Python (OpenCV, NumPy, Pandas, scikit-learn)
- MediaPipe (Pose Estimation)
- Arduino (LED Sync)
- GitHub for version control

## Future Work

- Deploy model for real-time feedback via mobile/web app
- Incorporate wrist, hand, and ball spin tracking
- Add Jetson Nano-based edge AI deployment
- Train reinforcement learning model to suggest form corrections



## MISC

git push origin main
git push lab main

