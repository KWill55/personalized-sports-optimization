flowchart LR

%% ===== Calibration Layer =====
subgraph Calibration
    Cal1[Camera Intrinsics]
    Cal2[Stereo Extrinsics]
    Cal3[Rim Geometry / Scale]
end

%% ===== Player Lane =====
subgraph Player_Kinematics
    V1[Raw Stereo Videos]
    V2[Synced Stereo Videos]
    K2D[2D Keypoints]
    K3D[3D Keypoints]
    A3D[3D Angles]
    K3D_C[3D Keypoints Cropped]
    A3D_C[3D Angles Cropped]
end

%% ===== Ball Lane =====
subgraph Ball_Kinematics
    BVID[Ball Cam Video]
    BTRJ[Ball Trajectory]
end

%% ===== Alignment & Events =====
subgraph Temporal_Logic
    SHIFT[Alignment Shift Table]
    PHASES[Free Throw Phases]
    OUTCOME[Shot Outcome]
end

%% ===== Flow =====
V1 --> V2 --> K2D --> K3D --> A3D --> K3D_C --> A3D_C --> PHASES
BVID --> BTRJ --> PHASES
SHIFT --> PHASES
PHASES --> OUTCOME

%% ===== Calibration Dependencies =====
Cal1 --> K2D
Cal1 --> K3D
Cal2 --> K3D
Cal3 --> BTRJ
