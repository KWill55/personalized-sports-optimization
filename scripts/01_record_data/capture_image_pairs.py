import cv2 as cv
import os

save_dir_left = "calib_images/left"
save_dir_right = "calib_images/right"
os.makedirs(save_dir_left, exist_ok=True)
os.makedirs(save_dir_right, exist_ok=True)

capL = cv.VideoCapture(0)
capR = cv.VideoCapture(1)

frame_id = 0

print("Press SPACE to capture image pair. Press ESC to quit.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Failed to grab frames.")
        break

    # Show live preview
    cv.imshow("Left", frameL)
    cv.imshow("Right", frameR)

    key = cv.waitKey(1)
    if key % 256 == 27:  # ESC to quit
        break
    elif key % 256 == 32:  # SPACE to capture
        fnameL = os.path.join(save_dir_left, f"left_{frame_id:02}.jpg")
        fnameR = os.path.join(save_dir_right, f"right_{frame_id:02}.jpg")
        cv.imwrite(fnameL, frameL)
        cv.imwrite(fnameR, frameR)
        print(f"Saved pair {frame_id}")
        frame_id += 1

capL.release()
capR.release()
cv.destroyAllWindows()
