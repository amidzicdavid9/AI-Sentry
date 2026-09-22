import cv2 as cv
from ultralytics import YOLO
import os
import numpy as np
from datetime import datetime
import time
import pygame

model = YOLO("yolov8n.pt")
existing_ids = set()
fourcc = cv.VideoWriter_fourcc(*'XVID')
pygame.mixer.init()

def captureLiveVideo():
    captureVideo = cv.VideoCapture(0, cv.CAP_V4L2)
    captureVideo.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'YUYV'))
    captureVideo.set(cv.CAP_PROP_FPS, 25)
    captureVideo.set(cv.CAP_PROP_BUFFERSIZE, 1)
    if not captureVideo.isOpened():
        exit()
    is_recording = False
    start_time = 0
    rec_clip = None
    object_in_frame_before = False
    while True:
        exist, frame = captureVideo.read()
        if exist:
            results = model.track(frame, classes=[0,2,5],persist=True, verbose=False)
            annotated_frame = results[0].plot()
            currently_count_object = len(results[0].boxes) > 0
            if currently_count_object and not object_in_frame_before:
                saveFrame(annotated_frame)
            object_in_frame_before = currently_count_object
            if results[0].boxes and results[0].boxes.id is not None:
                current_ids = results[0].boxes.id.int().cpu().tolist()
                for person_id in current_ids:
                    if person_id not in existing_ids:
                        existing_ids.add(person_id)
                        saveFrame(annotated_frame)
                        if not is_recording:
                            is_recording = True
                            pygame.mixer.music.load(f"Sentry-Alerts/Sounds/alarm_sound.mp3")
                            start_time = time.time()
                            pygame.mixer.music.play()
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            width = int(captureVideo.get(cv.CAP_PROP_FRAME_WIDTH))
                            height = int(captureVideo.get(cv.CAP_PROP_FRAME_HEIGHT))
                            rec_clip = cv.VideoWriter(f"Sentry-Alerts/Videos/REC_{timestamp}.avi", fourcc, 25.0, (width, height))
            if is_recording and rec_clip is not None:
                rec_clip.write(frame)
                if time.time() - start_time >= 60:
                    is_recording = False
                    pygame.mixer.music.stop()
                    rec_clip.release()
                    rec_clip = None
            cv.imshow("Live Video", annotated_frame)
        if cv.waitKey(1) == ord("q"):
            break
    captureVideo.release()
    cv.destroyAllWindows()


def saveFrame(annotated_frame):
    cwd = os.getcwd()
    folderPath = os.path.join(cwd, f"Sentry-Alerts/Images")
    timestamp = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
    filename = f"ALERT_{timestamp}.jpg"
    outPath = os.path.join(folderPath, filename)
    cv.imwrite(outPath, annotated_frame)


if __name__ == "__main__":
    captureLiveVideo()