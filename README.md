#  Real-Time Facial Emotion Detector

A real-time facial emotion detection system built with Python, OpenCV and DeepFace deep learning library. The system detects 7 human emotions from a live webcam feed and displays a full analytics panel with live percentage bars, emotion timeline graph, and auto screenshot capability.

---

##  Features
- Detects 7 emotions in real time: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- Live analytics panel showing all 7 emotions with percentage confidence bars
- Emotion timeline graph tracking mood changes over time
- Stability filter — emotion only changes after 4 consistent frames (no flickering)
- Multithreaded architecture — analysis runs in background for smooth performance
- Auto screenshot saved when strong happy emotion is detected (>80% confidence)
- Manual screenshot with S key
- FPS counter displayed live

---

##  Technologies Used
- Python 3.11
- OpenCV — webcam capture, face box drawing, UI rendering
- DeepFace — deep learning based emotion recognition model
- TF-Keras — backend for DeepFace model
- NumPy — frame and array processing
- Threading — background analysis for high FPS
- Collections (deque) — smoothing buffer for stable predictions

---

##  Installation

Make sure you have Python 3.11 installed. Then install all required libraries:

```bash
pip install opencv-python
pip install deepface
pip install tf-keras
pip install numpy
pip install fer
```

---

##  How to Run

```bash
python emotion_detector.py
```

The webcam will open automatically. On first run, DeepFace will download the emotion model (~6MB) — this happens only once.

---

##  Controls

| Key | Action |
|-----|--------|
| S | Save screenshot manually |
| Q | Quit the application |

---

##  Output
- Screenshots are automatically saved in the `screenshots/` folder
- Auto saved when happy emotion confidence exceeds 80%

---

##  Project Status
 Complete and working

---

##  Author
**Chandana R**  
B.E. Computer Science and Engineering — Garden City University, Bengaluru  
GitHub: [@chandanar8126](https://github.com/chandanar8126)
