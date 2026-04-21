import cv2
import numpy as np
from deepface import DeepFace
import os
from datetime import datetime
from collections import deque
import threading
import time

# ---------- SETUP ----------
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

EMOTION_COLORS = {
    "happy":    (0, 255, 128),
    "sad":      (255, 100, 50),
    "angry":    (0, 0, 255),
    "surprise": (0, 200, 255),
    "fear":     (180, 0, 255),
    "disgust":  (0, 180, 80),
    "neutral":  (200, 200, 200)
}

EMOTION_LABEL = {
    "happy":    "HAPPY   :)",
    "sad":      "SAD     :(",
    "angry":    "ANGRY  >:(",
    "surprise": "SURPRISE :O",
    "fear":     "FEAR    :/",
    "disgust":  "DISGUST x(",
    "neutral":  "NEUTRAL  :|"
}

# Smoothing + stability
emotion_buffer  = deque(maxlen=8)
history         = deque(maxlen=120)
stable_emotion  = "neutral"
stable_count    = 0
prev_dominant   = "neutral"

# Threading
analysis_result  = None
face_region      = None
lock             = threading.Lock()
analyzing        = False

# Misc
frame_count         = 0
screenshot_cooldown = 0
fps                 = 0
prev_time           = time.time()
smoothed            = {}

# ---------- BACKGROUND THREAD ----------
def analyze_frame(frame):
    global analysis_result, face_region, analyzing
    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'
        )
        with lock:
            analysis_result = result[0]['emotion']
            r = result[0]['region']
            face_region = (r['x'], r['y'], r['w'], r['h'])
    except:
        pass
    finally:
        analyzing = False

print("Emotion Detector — Final Version")
print("Press S to save screenshot | Press Q to quit")

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # Launch analysis every 3 frames only
    if frame_count % 3 == 0 and not analyzing:
        analyzing = True
        threading.Thread(
            target=analyze_frame,
            args=(frame.copy(),),
            daemon=True
        ).start()

    # Get latest result safely
    with lock:
        current_result = analysis_result
        current_region = face_region

    dominant = stable_emotion

    if current_result:
        emotion_buffer.append(current_result)

        smoothed = {
            emo: sum(f[emo] for f in emotion_buffer) / len(emotion_buffer)
            for emo in current_result
        }

        raw_dominant = max(smoothed, key=smoothed.get)

        if smoothed[raw_dominant] > 15:
            if raw_dominant == prev_dominant:
                stable_count += 1
            else:
                stable_count = 0
                prev_dominant = raw_dominant

            if stable_count >= 4:
                stable_emotion = raw_dominant

        dominant = stable_emotion
        history.append(dominant)

        # Auto screenshot on strong happy
        if screenshot_cooldown == 0:
            if dominant == "happy" and smoothed.get("happy", 0) > 80:
                fname = f"screenshots/happy_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Screenshot saved: {fname}")
                screenshot_cooldown = 90
        if screenshot_cooldown > 0:
            screenshot_cooldown -= 1

    # ---------- DRAW FACE BOX (uses cached region — no extra DeepFace call) ----------
    color = EMOTION_COLORS.get(dominant, (255, 255, 255))

    if current_region:
        rx, ry, rw, rh = current_region
        if rw > 10 and rh > 10:
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), color, 2)
            label = EMOTION_LABEL.get(dominant, dominant.upper())
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (rx, ry - lh - 18), (rx + lw + 10, ry), color, -1)
            cv2.putText(frame, label, (rx + 5, ry - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 0.001)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------- RIGHT PANEL ----------
    panel_w = 290
    canvas = np.zeros((h, w + panel_w, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (22, 22, 22)
    px = w + 15

    # Title
    cv2.putText(canvas, "EMOTION ANALYTICS", (px, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.line(canvas, (px, 45), (px + 255, 45), (70, 70, 70), 1)

    # Emotion bars
    if smoothed:
        bar_y = 75
        for emo, score in sorted(smoothed.items(), key=lambda x: x[1], reverse=True):
            c = EMOTION_COLORS.get(emo, (200, 200, 200))
            is_top = (emo == dominant)

            cv2.putText(canvas, emo.upper(), (px, bar_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 if is_top else 0.5,
                        c if is_top else (160, 160, 160),
                        2 if is_top else 1)

            cv2.rectangle(canvas, (px, bar_y + 4),
                          (px + 240, bar_y + 18), (55, 55, 55), -1)
            fill = int((score / 100) * 240)
            cv2.rectangle(canvas, (px, bar_y + 4),
                          (px + fill, bar_y + 18), c, -1)
            cv2.putText(canvas, f"{score:.1f}%",
                        (px + 244, bar_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            bar_y += 42

    # ---------- TIMELINE GRAPH ----------
    graph_top = h - 130
    graph_h   = 90
    graph_w   = 255
    emotion_list = list(EMOTION_COLORS.keys())

    cv2.rectangle(canvas,
                  (px, graph_top),
                  (px + graph_w, graph_top + graph_h),
                  (40, 40, 40), -1)
    cv2.putText(canvas, "Emotion Timeline",
                (px, graph_top - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Dominant emotion label above timeline
    cv2.putText(canvas, EMOTION_LABEL.get(dominant, dominant.upper()),
                (px + 80, graph_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                EMOTION_COLORS.get(dominant, (255, 255, 255)), 2)

    if len(history) > 1:
        hist = list(history)
        for i in range(1, len(hist)):
            x1 = px + int((i - 1) / 120 * graph_w)
            x2 = px + int(i / 120 * graph_w)
            idx1 = emotion_list.index(hist[i-1]) if hist[i-1] in emotion_list else 0
            idx2 = emotion_list.index(hist[i])   if hist[i]   in emotion_list else 0
            y1 = graph_top + graph_h - int((idx1 / 6) * graph_h) - 5
            y2 = graph_top + graph_h - int((idx2 / 6) * graph_h) - 5
            c  = EMOTION_COLORS.get(hist[i], (200, 200, 200))
            cv2.line(canvas, (x1, y1), (x2, y2), c, 1)

    # Controls
    cv2.putText(canvas, "S = screenshot   Q = quit",
                (px, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

    cv2.imshow("Emotion Detector  |  Chandana R", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"screenshots/manual_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(fname, frame)
        print(f"Screenshot saved: {fname}")

cap.release()
cv2.destroyAllWindows()
print("Session ended. Screenshots saved in /screenshots folder.")