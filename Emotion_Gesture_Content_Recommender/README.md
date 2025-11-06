## 🎭 AI-Powered Emotion & Gesture-Based Content Recommender

## 🧠 ABOUT THIS PROJECT ==>
- An intelligent social media engine that uses real-time emotion detection and hand gesture recognition via your webcam to recommend personalized content and stories based on your mood and interaction style.

- This application includes:
1. CNN-based Emotion Detection (FER2013 dataset)
2. ML-based Hand Gesture Recognition
3. Personalized Recommender System for videos/stories
4. AI-Powered Story Generator (using your mood as context)
5. Dynamic Vibe Score tracking your emotional journey
6. Session Analytics & Visualizations

- Ideal for applications in wellness tech, AI-enhanced social platforms, and emotion-aware personalization engines.

---

## ⚙ TECHNOLOGIES USED ==>
- Python
- OpenCV / MediaPipe (real-time face & hand landmark detection)
- TensorFlow / Keras (emotion classification model)
- Scikit-learn (gesture classification model)
- Matplotlib / Seaborn / Pandas (EDA & visual analytics)
- TextBlob / NLP (mood-based AI story generation)
- CSV / Pickle (data storage and logging)

---

## 📁 PROJECT FOLDER STRUCTURE ==>

main_folder/
│
├── data/
│   ├── emotion/
│   │   └── fer2013/                  # Raw dataset for training the emotion recognition model
│   ├── content_pool.csv              # Backup pool of content samples for recommendation system
│   ├── gesture_data.csv              # Landmark data extracted from gestures (used to train gesture model)
│   ├── mock_user_data.csv            # Dummy user data
│   ├── recommendation_content.csv    # Curated content mapped to emotions/gestures for personalized feed
│   └── session_logs.csv              # Logs of user sessions for behavior analysis and vibe score tracking
│
├── models/
│   ├── analyze_session.py            # Generates visual session analysis from logs (mood/vibe trends visualization)
│   ├── collect_gesture_data.py       # Script to collect gesture landmark data using MediaPipe (for training)
│   ├── emotion_detector.py           # Optional: Run real-time emotion detection independently
│   ├── emotion_model.h5              # Trained CNN model for emotion recognition (FER2013-based)
│   ├── emotion_model.py              # Script to train the CNN emotion recognition model
│   ├── gesture_classifier.py         # Landmark extraction & helper for gesture classification
│   ├── gesture_model.pkl             # Trained scikit-learn model for gesture recognition
│   ├── gesture_predictor.py          # Optional: Script to test gesture model in real-time (debugging)
│   ├── recommender.py                # Main recommendation logic using user mood/gesture as input
│   ├── story_generator.py            # Uses GPT to generate AI-driven interactive stories based on mood
│   ├── train_gesture_model.py        # Trains gesture recognition model on extracted landmarks
│   └── vibe_score.py                 # Calculates "Vibe Score" based on user behavior and engagement
│
├── utils/
│   ├── analytics_logger.py           # Logs session details, moods, actions for analytics
│   ├── data_loader.py                # Utility functions for loading models, CSVs, etc.
│   └── preprocess_emotion_data.py    # Preprocessing FER2013 data before training
│
├── main.py                           # Main application entry point — runs the real-time system
├── requirements.txt                  # Python dependencies list
└── README.md                         # Project overview, setup instructions, features, and usage guide

---

## 📝 WHAT EACH FILE DOES ==>
*main.py*
- Runs the full pipeline: camera opens → emotion + gesture prediction → content shown → session logged → analytics run.

*emotion_model.py*
- Trains CNN model using FER2013 emotion dataset.

*gesture_classifier.py / gesture_predictor.py*
- Load gesture model and predict hand gestures in real-time.

*recommender.py*
- Recommends videos/posts/stories based on current emotion and gesture.

*story_generator.py*
- AI story creator that builds short stories based on your current mood.

*vibe_score.py*
- Calculates and updates your "Vibe Score" based on emotion shifts.

*analyze_session.py*
- Reads session logs and plots mood/vibe trends using seaborn/matplotlib.

*analytics_logger.py*
- Saves each session’s timestamp, emotion, gesture, and recommendation.

*train_gesture_model.py*
- Trains ML model on collected hand landmark features.

*collect_gesture_data.py*
- Script to gather gesture data by showing labels during webcam capture.

---

## 🚀 HOW TO RUN THE PROJECT ==>

# Step 1: Move into the directory
cd D:\Emotion_Gesture_Content_Recommender
D:

# Step 2: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Full App
python main.py

### OTHER STEPS TO RUN [ run_them_after_installing_requirements! ] ==>>
python models/emotion_model.py               << TO TRAIN emotion model (FER2013, CNN)
python models/collect_gesture_data.py        << TO COLLECT hand landmarks for gestures (saves CSV)
python models/train_gesture_model.py         << TO TRAIN gesture classifier (scikit-learn model)
python models/emotion_detector.py            << TO RUN/TEST emotion detection in real-time (for testing only)
python models/gesture_classifier.py          << TO TEST trained gesture model on webcam
python models/gesture_predictor.py           << TO PREDICT gesture from saved CSV or sample input

---

## ✅ IMPROVEMENTS MADE ==>
- Integrated real-time dual-mode detection (emotion + gesture)
- Personalized mood-based recommender system
- AI-based story generation engine
- Logged full session data for mood progression analytics
- Added a dynamic Vibe Score for emotional well-being insights
- Modular structure for future integration of GPT agents or Web GUI

---

## 📌 TO DO / FUTURE ENHANCEMENTS ==>
- Build a Streamlit dashboard for content browsing and analytics
- Integrate audio sentiment along with visual mood detection
- Add user profiling for personalized long-term recommendations
- Introduce multi-language emotion support
- Gamify experience using badges based on vibe trends
- Use clustering or sequence models for next-mood prediction

---

## ✨ SAMPLE OUTPUT ==>
🎥 Webcam Activated
🙂 Detected Emotion: "Happy"
🤟 Detected Gesture: "Heart"
🎬 Recommendation: "Motivational video – You Can Do It!"
📖 Story Generated: "You’re unstoppable today, just like the sun pushing through the clouds..."
📊 Analytics: Vibe Score +2 | 4 gestures, 3 emotions recorded

---

## 📬 **CONTACT / COLLABS ==>
For questions or feedback, feel free to reach out!

---
