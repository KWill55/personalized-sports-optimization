from calibration.calibration_node import CalibrationNode
from calibration.calibration_capture_gui import CalibrationCaptureGui

from recording.recording_node import RecordingNode
from recording.recording_capture_gui import RecordingCaptureGui

from player_tracking.pose_2d_node import Pose2dNode
from player_tracking.pose_3d_node import Pose3dNode

from ball_tracking.ball_detection_node import BallDetectionNode

from preprocessing.alignment_node import AlignmentNode
from preprocessing.phases_node import PhasesNode

from evaluation.grading_node import GradingNode

"""
Purpose: orchestrate entire system

"""

def main():

    print("Welcome to the Freethrow Optimization System!\n")

    # =========================================================
    # Create Objects
    # =========================================================
    
    calibration_node = CalibrationNode()
    calibration_capture_gui = CalibrationCaptureGui()

    recording_node = RecordingNode()
    recording_capture_gui = RecordingCaptureGui()

    pose_2d_node = Pose2dNode()
    pose_3d_node = Pose3dNode()

    ball_detection_node = BallDetectionNode()

    alignement_node = AlignmentNode()
    phases_node = PhasesNode()

    grading_node = GradingNode()


    # =========================================================
    # Repl loop to allow user selection for operation repeatedly
    # =========================================================

    try:
        while True:
            print("perform task requested by user")
            # TODO maybe give options for what the user wants to do or display what data is available
            #   user might only want to open GUIs for instance and they might want to only record more data  
            # switch type statement to see what the user wants to do repeatedly like a repl loop?
            # or boolean values at the top that will say what the user wants to do 
            # probably make a node like UI or UX node for this 

    except KeyboardInterrupt:
        pass
    
    finally:
        try: calibration_node.close()
        except Exception: pass
        try: calibration_capture_gui.close()
        except Exception: pass
        try: recording_node.close()
        except Exception: pass
        try: recording_capture_gui.close()
        except Exception: pass
        try: pose_2d_node.close()
        except Exception: pass
        try: pose_3d_node.close()
        except Exception: pass
        try: ball_detection_node.close()
        except Exception: pass
        try: alignement_node.close()
        except Exception: pass
        try: phases_node.close()
        except Exception: pass
        try: grading_node.close()
        except Exception: pass
    
if __name__ == "__main__":
    main()





