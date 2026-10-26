# Emotion-Aware Intelligent Robot with Computer Vision and Conversational AI

## 1. Abstract
An emotion-aware robot that uses computer vision and deep learning to classify visible facial
expressions in real time. The detected expression is passed as context to a local conversational AI
(Ollama), which enables adaptive text and voice interaction. Guests get temporary sessions; registered
users get saved history, preferences and robot memory. Analytics, statistics and reports are built only
from stored data.

## 2. Problem statement
Most chat assistants ignore the user's visible state, and most emotion demos cannot hold a conversation.
This project joins the two while staying honest: it reports a *detected facial expression*, never a claim
about what a person truly feels.

## 3. Objectives
- Real-time face detection and 7-class expression recognition on a normal Windows laptop.
- A stable robot behaviour (smoothing, confidence threshold, cooldown) instead of flickering reactions.
- Local conversational AI with Ollama, no paid APIs, camera frames never leave the computer.
- Guest mode, user accounts, separated per-user data, explicit user-controlled robot memory.

## 4. Architecture
```text
Camera -> YuNet face detection -> preprocessing -> Mini-Xception -> 7 probabilities
  -> temporal smoothing -> confidence filter -> frame stabilisation -> Decision engine
  -> Robot state (mood, action, style) -> Ollama (only on user messages) -> text / voice reply
  -> temporary store (guest) or SQLite (user) -> graphs, statistics, reports
```

## 5. Technologies
Python 3.11, Flask, OpenCV (YuNet), Keras/TensorFlow (Mini-Xception, FER-2013), Ollama, SQLite,
pyttsx3 (offline text-to-speech), browser Web Speech API (voice input), HTML/CSS/JavaScript with
hand-drawn canvas charts (no internet needed for charts).

## 6. Project structure
```text
emotion_robot/
├── app.py               Flask app: routes, auth, wiring
├── auth.py              registration checks, password hashing
├── services.py          per-session runtime (engine, chat memory, temporary store), settings validation
├── vision/              face_detector.py, emotion_classifier.py, preprocessing.py, errors.py
├── robot/               decision_engine.py, robot_state.py, response_manager.py, speech.py
├── ai/ollama_client.py  local chat AI
├── database/            db.py (SQLite), session_store.py (guest memory), analytics.py (statistics)
├── models/              emotion_model.h5 and face_detector/*.onnx (downloaded by script)
├── templates/ static/   pages, css, js
├── scripts/download_models.py
├── evaluation/evaluate_model.py
├── tests/
└── data/ reports/ dataset/
```

## 7. Pages
Landing, Login/Sign up, Guest mode, Dashboard, Emotion Detection, AI Chat, Emotion Graphs, Statistics,
Reports, My Memory, Profile, Settings, How It Works, Logout.

## 8. Computer vision pipeline and emotion recognition
1. The browser sends a 640 px JPEG frame every 300-800 ms to `/detect`.
2. YuNet returns face boxes. 3. Each face is cropped, grayscale, resized to 64x64, scaled to [-1, 1].
4. Mini-Xception returns 7 probabilities (angry, disgust, fear, happy, sad, surprise, neutral).
5. The boxes and labels are drawn on a canvas over the live video. Multiple faces are shown; the robot
   reacts to the largest one. No identity recognition is done.

## 9. Robot decision-making
`robot/decision_engine.py`: exponential smoothing of probabilities, a confidence threshold (default 35%),
3 consistent frames before the robot changes state, a speech cooldown (8 s), no speech for neutral, rotating
reaction lines to avoid repetition. It outputs mood, action, response style and whether to speak.
It never calls Ollama for camera frames.

## 10. Ollama integration
`ai/ollama_client.py` talks to `http://127.0.0.1:11434`. **Normal AI** answers normally. **Emotion-Aware AI**
adds "Detected facial expression: X (confidence)" to the system prompt as context only. Remembered
preferences are added as preferences. If Ollama is off, a small built-in reply set is used and the UI says so.

## 11. Database design
`users`, `sessions`, `emotion_records`, `conversations`, `robot_memory`, `user_preferences`, `reports`.
All rows carry `user_id` (foreign key, cascade delete) and every query filters by it. Passwords are stored as
PBKDF2-SHA256 hashes. Guests never touch the database: their data is an in-memory store that expires after
2 hours of inactivity or when the server restarts.

## 12. Installation (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts\download_models.py
python app.py
```
Open http://127.0.0.1:5000. If a model download fails, the script prints where to download it by hand.

## 13. Ollama setup
Install Ollama, then `ollama pull llama3.2` (or any model). Keep it running. Pick a model in
Settings, or set `$env:OLLAMA_MODEL="llama3.2"` before `python app.py`.

## 14. Using it
- **Guest:** click *Continue as guest*. History, memory, graphs and reports are for this session only.
- **Account:** *Login / Sign up*. History, chats, reports, memory and settings are saved per user.
- **Voice:** in Chrome or Edge, press the microphone in AI Chat. Speech recognition there uses the browser
  vendor's online service, so it needs internet. Camera frames stay local.
- **Memory:** say "remember that I prefer short answers" or add it on My Memory. Nothing is inferred automatically.
- **Database setup:** automatic. `data/robot.db` is created on first run. Delete it to reset everything.

## 15. Testing
`python -m pip install -r requirements-dev.txt` then `python -m pytest`. The tests cover the decision engine,
database isolation between users, password hashing, the Ollama client, guest/user flows, CSRF, memory,
settings, account deletion and missing-model handling. Vision modules need the real model files and were not
covered by automated tests.

Manual checklist: guest flow, sign up, log out and in, camera on/off, one and two faces, covered camera,
Ollama on/off, microphone allowed/blocked, remember command, delete memory, download report, delete data.

## 16. Evaluation
```powershell
python -m pip install -r requirements-dev.txt
python evaluation\evaluate_model.py --csv dataset\fer2013.csv --split PrivateTest
```
Writes real accuracy, macro precision/recall/F1, per-emotion metrics and the confusion matrix to
`reports/model_metrics.json`. No result is hardcoded. It needs the model and a legally obtained `fer2013.csv`.

## 17. Limitations
- About 66% accuracy on FER-2013; webcam accuracy is usually lower. Disgust and fear are the weakest classes.
- Expression is not emotion; results differ across people, lighting and culture.
- Single-machine app: login is basic (no email verification, no rate limiting), guest data is lost on restart.
- Voice input depends on Chrome/Edge and internet.
- The camera runs only on the Emotion Detection page or the chat page's camera card.

## 18. Future enhancements
Physical robot or ROS integration, multimodal (voice) emotion recognition, person tracking, a fine-tuned model
on webcam data, richer personalization, email verification and rate limiting.

## 19. Viva summary
"The browser sends frames to Flask. YuNet finds faces and Mini-Xception outputs 7 expression probabilities.
A decision engine smooths them and changes the robot's state only after consistent frames. When the user
chats, Ollama writes the reply and the detected expression is only context. Guests are temporary; users are
stored in SQLite and separated by user ID, with hashed passwords."
