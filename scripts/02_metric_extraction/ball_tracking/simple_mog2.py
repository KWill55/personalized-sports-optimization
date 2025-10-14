import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# -----------------------------------------------
# Helper: pick a video file using a dialog
# -----------------------------------------------


# class BallTrackingGui:
    
def choose_file(title):
    root = tk.Tk()
    root.withdraw()   # hide main window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.update()
    root.destroy()
    return file_path



class BackgroundSubtractor():
    
    def __init__(self):
        self.lr = 0 # learning rate 
        self.detectShadows = False
        #TODO set more parameters here 
        
        self.bgsub = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=16, detectShadows=self.detectShadows)

    def train_background(self):
        """
        trains mog2 on background-only clip 
        """

        # choose clip that has background only 
        bg_clip = choose_file(title="Select background-only clip")
        if not bg_clip:
            raise SystemExit("No background clip selected.")
        
        # train mog2 model on background clip 
        print("[INFO] Training background model...")
        self.lr = .02
        cap = cv2.VideoCapture(bg_clip)
        frames_trained = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert video to grayscale 
            gray = cv2.GaussianBlur(gray, (5, 5), 0) # apply averaging kernal to smooth mask 
            self.bgsub.apply(gray, learningRate=self.lr)   # positive learning rate while training
            frames_trained += 1
            cv2.imshow("Training background", frame)
            if cv2.waitKey(10) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Background model trained on {frames_trained} frames.\n")


    def apply_model_to_clip(self):
        """
        applies mog2 model to a chosen free throw clip 
        """
        
        # choose free throw clip to apply mog2 model to 
        throw_clip = choose_file("Select free-throw clip")
        if not throw_clip:
            raise SystemExit("No free-throw clip selected.")
        

        # apply mog2 model to chosen clip 
        cap2 = cv2.VideoCapture(throw_clip)
        print("[INFO] Applying trained model (learningRate=0)... Press ESC to quit.")

        self.lr = 0 # no training anymore 
        while True:
            ok, frame = cap2.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert video to grayscale
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            mask = self.bgsub.apply(gray, learningRate=self.lr)   # apply mog2 mask to video 
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k) # erode
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k) # dilate 

            # show original and mask
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((frame, mask_color))
            cv2.imshow("Original | MOG2 Mask", combined)

            if cv2.waitKey(30) & 0xFF == 27:  # ESC to quit
                break

        cap2.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


def main():
    subtractor = BackgroundSubtractor()
    subtractor.train_background()
    subtractor.apply_model_to_clip()


if __name__ == "__main__": 
    main()

