---
config:
  theme: redux
---
flowchart LR
subgraph Calibration
    INTRIN[Camera Intrinsics]
    EXTRIN[Stereo Extrinsics]
    CALIB[Calibration Parameters]
    INTRIN_CALIB_IMAGES[Intrinsic Calibration Images]
    EXTRIN_CALIB_IMAGES[Extrinsic Calibration Images]

end
subgraph Player Kinematics
    V1[Raw Stereo Videos]
    V2[Synced Stereo Videos]
    K2D[2D Keypoints]
    K3D[3D Keypoints]
    PLAYER_CROPPED[Cropped Player Keypoints]
end
subgraph Ball Tracking
    BVID[Ball Cam Video]
    BDET[Ball Detections]
    BDET_C[Cropped Ball Detections]

end
subgraph Alignment
    SHIFT[Alignment Shift Table from player keypoints: Time Mapping]
end
subgraph Freethrow Phases
    PHASES[Free Throw Phases]
end
subgraph Shot Quality Features
    PLAYER_ALIGNED[Aligned Player Keypoints and computed player signals: angles, velocity, etc]
    BALL_ALIGNED[Aligned Ball Detection and computed ball signals: trajecory, velocity, etc]
end
subgraph Shot Quality Label
    OUTCOME[Shot Outcome]
end
subgraph Grader: Internal Consistency 
    PLAYER_CONSIST[Internal Consistency Grading: Ball, Player, and combined signals]
end
subgraph Grader: Golden Reference
    TEMPLATE[Golden Template]
    PLAYER_DEV[PLayer/Ball Deviation from Golden Template: Keypoints, Angle]
end
subgraph Athlete Evaluation
    SESSION_REPORT[Per-shot score / Session summary / coaching cues]
end
INTRIN_CALIB_IMAGES --> INTRIN
EXTRIN_CALIB_IMAGES --> EXTRIN
INTRIN --> CALIB
EXTRIN --> CALIB
CALIB --> K3D

BVID --> BDET
BDET --> BDET_C


PLAYER_CROPPED --> SHIFT

K3D --> PHASES
BDET --> PHASES
PHASES --> PLAYER_CROPPED
PHASES --> BDET_C

SHIFT --> PLAYER_ALIGNED
PLAYER_CROPPED --> PLAYER_ALIGNED

V1 --> V2 --> K2D --> K3D --> PLAYER_CROPPED



BDET_C --> BALL_ALIGNED
SHIFT --> BALL_ALIGNED

BALL_ALIGNED --> PLAYER_CONSIST
PLAYER_ALIGNED --> PLAYER_CONSIST


BALL_ALIGNED --> PLAYER_DEV
PLAYER_ALIGNED --> PLAYER_DEV
TEMPLATE --> PLAYER_DEV


OUTCOME --> SESSION_REPORT
PLAYER_CONSIST --> SESSION_REPORT
PLAYER_DEV --> SESSION_REPORT

BVID --> OUTCOME