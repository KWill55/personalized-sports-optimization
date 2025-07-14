"""
Title: combine_player_feeds.py

Description: 
    This module's purpose is to combine player tracking feeds from two cameras into a single video feed.
    It combines two 640x640 videos into a single 1280x640 video for player tracking. 
    The combined video is saved in a structurured directory. 

Inputs:
    - Left and right player tracking videos (each 640x640)

Usage:
    - Running the script combines the two player feeds into a single video feed. 

Outputs
    - 1280x640 videos for player tracking:
        - side by side stereo feeds (each 640x640)
"""



video_dirs = {
    "player": session_dir / "videos" / "player_tracking" / "raw",
    "ball": session_dir / "videos" / "ball_tracking" / "raw" 
}