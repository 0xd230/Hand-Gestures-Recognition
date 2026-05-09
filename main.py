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

matLayer = 0
results = {}

for file in Path(".").glob("*.svd*"):
    if file.is_file():
        results[str(matLayer)] = (file.name, 0)
        matLayer += 1

handMat = numpy.ones((21, 2))
refMat = numpy.ones((matLayer, 21, 2))

currlayer = -1
for file in Path(".").glob("*.svd*"):
    if file.is_file():
        currlayer += 1
        print(file.name)
        with open(file.name, "r") as f:
            next(f)
            index = 0
            for line in f:
                coor = [p.strip() for p in line.split(' ')]
                refMat[currlayer][index][0] = coor[0]
                refMat[currlayer][index][1] = coor[1]
                index += 1

def process_result(result: handLandmarkerRes, outIMG: mp.Image, timestamp_ms: int):
    #process hand gestures here
    if (handLM := result.hand_landmarks):
        for i in range(matLayer):
            for j in range(21):
                if j == 0:
                    handMat[j][0] = handLM[0][j].x
                    handMat[j][1] = handLM[0][j].y
                else:
                    handMat[j][0] = handLM[0][j].x - handMat[0][0]
                    handMat[j][1] = handLM[0][j].y - handMat[0][1]

options = handLandmarkerOpt(base_options=baseOpt(model), running_mode = runningMode.LIVE_STREAM, result_callback=process_result, num_hands=1)

textindex = 0
filename = "text0.txt"
filepath = Path(filename)
while filepath.is_file():
    textindex += 1
    filename = "text" + str(textindex) + ".txt"
    filepath = Path(filename)

with handLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        ret, frame = cam.read()
        dst = cv.flip(frame, 1)
        WINDH, WINDW, _ = dst.shape

        cosApprox = 0
        euDist = 0
        euApprox = 0

        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(mp.ImageFormat.SRGB, cv.cvtColor(dst, cv.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, timestamp_ms)


        for i in range(21):
            if i == 0:
                cv.circle(dst, (round((handMat[i][0]) * WINDW), round((handMat[i][1]) * WINDH)), 15, (255, 255, 255), cv.FILLED)
            else:
                cv.circle(dst, (round((handMat[i][0] + handMat[0][0]) * WINDW), round((handMat[i][1] + handMat[0][1]) * WINDH)), 15, (255, 255, 255), cv.FILLED)

        vars = numpy.zeros(matLayer)
        count = -1
        for i in range(matLayer):
            for j in range(21):
                if j == 0:
                    continue
                cosApprox += (numpy.dot(handMat[j], refMat[i][j])) / (numpy.sqrt(handMat[j][0] ** 2 + handMat[j][1] ** 2) * numpy.sqrt(refMat[i][j][0] ** 2 + refMat[i][j][1] ** 2))
                euDist = numpy.sqrt((handMat[j][0] / numpy.sqrt(handMat[9][0] ** 2 + handMat[9][1] ** 2) - refMat[i][j][0] / numpy.sqrt(refMat[i][9][0] ** 2 + refMat[i][9][1] ** 2)) ** 2 + (handMat[j][1] / numpy.sqrt(handMat[9][0] ** 2 + handMat[9][1] ** 2) - refMat[i][j][1] / numpy.sqrt(refMat[i][9][0] ** 2 + refMat[i][9][1] ** 2)) ** 2)
                euApprox += 1 / (1 + euDist)

            count += 1
            cosApprox /= 20
            euApprox /= 20
            name = results[str(count)][0]
            results[str(count)] = (name, (cosApprox + euApprox) / 2)

        #results = {index, (filename, approxvalue)}
        resultTuple = max(results.values(), key=lambda item: item[1])
        resultName, resultVal = resultTuple

        cv.putText(dst, resultName, (40, 120), cv.FONT_HERSHEY_DUPLEX, 4, (0, 0, 0), 14, cv.LINE_AA)
        cv.putText(dst, resultName, (40, 120), cv.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0), 5, cv.LINE_AA)

        cv.imshow('frame', dst)
        key = cv.waitKey(1) & 0xFF

        file_content = ""
        new_file_content = ""
        if key == ord('q'):
            print("Quitting :(")
            break
        elif key == ord('t'):
            with open(filename, "a") as file:
                print("i wote \"" + resultName[:-4] + "\" to " + filename + "nya~ OwO")
                file.write(resultName[:-4])
        elif key == ord('b'):
                print("backspace nya~! :3")
                with open(filename, "r") as file:
                    file_content = file.read()
                
                new_file_content = file_content[:-1]

                with open(filename, "w") as file:
                    file.write(new_file_content)

cam.release()
cv.destroyAllWindows()