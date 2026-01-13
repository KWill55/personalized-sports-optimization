from graphviz import Digraph

dot = Digraph("FreeThrowPipeline", comment="Free Throw Data Contract")

# Calibration
with dot.subgraph(name="cluster_calib") as c:
    c.attr(label="Calibration")
    c.node("Cal1", "Camera Intrinsics")
    c.node("Cal2", "Stereo Extrinsics")
    c.node("Cal3", "Rim Geometry / Scale")

# Player
with dot.subgraph(name="cluster_player") as p:
    p.attr(label="Player Kinematics")
    p.node("V1", "Raw Stereo Videos")
    p.node("V2", "Synced Stereo Videos")
    p.node("K2D", "2D Keypoints")
    p.node("K3D", "3D Keypoints")
    p.node("A3D", "3D Angles")
    p.node("K3DC", "3D Keypoints Cropped")
    p.node("A3DC", "3D Angles Cropped")

# Ball
with dot.subgraph(name="cluster_ball") as b:
    b.attr(label="Ball Kinematics")
    b.node("BVID", "Ball Cam Video")
    b.node("BTRJ", "Ball Trajectory")

# Events
with dot.subgraph(name="cluster_events") as e:
    e.attr(label="Temporal Logic")
    e.node("SHIFT", "Alignment Shift Table")
    e.node("PHASES", "Free Throw Phases")
    e.node("OUTCOME", "Shot Outcome")

# Main pipeline
dot.edges([
    ("V1", "V2"),
    ("V2", "K2D"),
    ("K2D", "K3D"),
    ("K3D", "A3D"),
    ("A3D", "K3DC"),
    ("K3DC", "A3DC"),
    ("A3DC", "PHASES"),
    ("BVID", "BTRJ"),
    ("BTRJ", "PHASES"),
    ("SHIFT", "PHASES"),
    ("PHASES", "OUTCOME")
])

# Calibration dependencies
dot.edges([
    ("Cal1", "K2D"),
    ("Cal1", "K3D"),
    ("Cal2", "K3D"),
    ("Cal3", "BTRJ")
])

dot.render("free_throw_data_contract", view=True)
