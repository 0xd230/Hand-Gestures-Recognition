import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy
import time
from pathlib import Path

cam = cv.VideoCapture(0)

model = 'hand_landmarker.task'

baseOpt = mp.tasks.BaseOptions
handLandmarker = mp.tasks.vision.HandLandmarker
handLandmarkerOpt = mp.tasks.vision.HandLandmarkerOptions
handLandmarkerRes = mp.tasks.vision.HandLandmarkerResult
runningMode = mp.tasks.vision.RunningMode

handMat = numpy.ones((21, 2))
handMatTot = numpy.zeros((21, 2))
ticks = 0
recording = False

def process_result(result: handLandmarkerRes, outIMG: mp.Image, timestamp_ms: int):
    #process hand gestures here
    if (handLM := result.hand_landmarks):
        for i in range(21):
            if i == 0:
                handMat[i][0] = handLM[0][i].x
                handMat[i][1] = handLM[0][i].y
                if recording:
                    handMatTot[i][0] += handLM[0][i].x
                    handMatTot[i][1] += handLM[0][i].y
            else:
                handMat[i][0] = handLM[0][i].x - handMat[0][0]
                handMat[i][1] = handLM[0][i].y - handMat[0][1]
                if recording:
                    handMatTot[i][0] += handLM[0][i].x - handMat[0][0]
                    handMatTot[i][1] += handLM[0][i].y - handMat[0][1]

options = handLandmarkerOpt(base_options=baseOpt(model), running_mode = runningMode.LIVE_STREAM, result_callback=process_result, num_hands=1)

with handLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        ret, frame = cam.read()
        dst = cv.flip(frame, 1)
        WINDH, WINDW, _ = dst.shape

        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(mp.ImageFormat.SRGB, cv.cvtColor(dst, cv.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, timestamp_ms)

        for i in range(21):
            if i == 0:
                cv.circle(dst, (round((handMat[i][0]) * WINDW), round((handMat[i][1]) * WINDH)), 15, (255, 255, 255), cv.FILLED)
            else:
                cv.circle(dst, (round((handMat[i][0] + handMat[0][0]) * WINDW), round((handMat[i][1] + handMat[0][1]) * WINDH)), 15, (255, 255, 255), cv.FILLED)

        if recording:
            print("recording")
            ticks += 1
            cv.circle(dst, (WINDW - 100, 100), 50, (0, 0, 0), cv.FILLED)
            cv.circle(dst, (WINDW - 100, 100), 40, (0, 255, 0), cv.FILLED)
        else:
            print("not recording")

        cv.imshow('frame', dst)
        key = cv.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
        elif key == ord('q'):
            break

index = 0
filepath = Path("data" + str(index) + ".svd")
while filepath.is_file():
    index += 1
    filepath = Path("data" + str(index) + ".svd")

with open("data" + str(index) + ".svd", "a") as file:
    for i in range(21):
        file.write("\n" + str(handMatTot[i][0] / ticks) + " " + str(handMatTot[i][1] / ticks))

print("Saved nyaw~! :3")

cam.release()
cv.destroyAllWindows()