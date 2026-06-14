# 🎵 Music Recommendation Based on Facial Emotion Recognition

## 📌 Overview

Music has a significant impact on human emotions and well-being. This project combines **Computer Vision**, **Deep Learning**, and **Recommendation Systems** to provide personalized music recommendations based on a user's facial emotion.

The system detects a user's facial expression from an uploaded image using a **ResNet18-based Facial Emotion Recognition model** and recommends songs that match the detected emotional state.

---

## 🚀 Features

* Facial Emotion Recognition using Deep Learning
* Emotion Detection from Uploaded Face Images
* Music Mood Mapping Engine
* Content-Based Music Recommendation System
* FastAPI Backend for Prediction Services
* Streamlit Frontend for Interactive User Experience
* Real-Time Emotion-to-Music Recommendation Pipeline

---

## 🏗️ System Architecture

```text
User Face Image
       │
       ▼
ResNet18 Emotion Detection Model
       │
       ▼
Detected Emotion
       │
       ▼
Emotion → Music Mood Mapping
       │
       ▼
Content-Based Recommender
       │
       ▼
Top Recommended Songs
```

---

## 🧠 Technologies Used

### Frontend

* Streamlit

### Backend

* FastAPI
* Uvicorn

### Deep Learning

* PyTorch
* TorchVision

### Data Processing

* Pandas
* NumPy
* Scikit-Learn

### Visualization

* Matplotlib

---

## 📂 Dataset Information

### Facial Emotion Dataset

FER-2013 Dataset

Emotions:

* Happy
* Sad
* Angry
* Fear
* Disgust
* Surprise
* Neutral

### Music Dataset

Spotify Mood Dataset

Attributes:

* Danceability
* Energy
* Acousticness
* Instrumentalness
* Valence
* Tempo
* Loudness
* Speechiness
* Popularity
* Mood Labels

Music Moods:

* Happy
* Sad
* Energetic
* Calm

---

## 🤖 Emotion Recognition Model

### Model

ResNet18 (Transfer Learning)

### Input

* Face Image
* 224 × 224 Resolution

### Output

* Happy
* Sad
* Angry
* Fear
* Disgust
* Surprise
* Neutral

## 🎼 Recommendation Engine

The recommendation engine performs:

1. Emotion Detection
2. Emotion-to-Mood Mapping
3. Mood-Based Song Filtering
4. Content-Based Similarity Ranking
5. Top-N Song Recommendation

### Emotion Mapping

| Emotion  | Music Mood |
| -------- | ---------- |
| Happy    | Happy      |
| Sad      | Sad        |
| Angry    | Calm       |
| Fear     | Calm       |
| Disgust  | Calm       |
| Neutral  | Calm       |
| Surprise | Energetic  |

---

## 📁 Project Structure

```text
music_emotion_app/

├── backend/
│   ├── main.py
│   ├── best_resnet18_emotion_model.pth
│   ├── data_moods.csv
│
├── frontend/
│   ├── app.py
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/prasad2023222/music-recommendation-facial-emotion.git
cd music-recommendation-facial-emotion
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend URL:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## ▶️ Run Frontend

```bash
cd frontend
streamlit run app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## 📊 Workflow

1. User uploads a facial image.
2. FastAPI receives the image.
3. ResNet18 predicts the facial emotion.
4. Emotion is mapped to a music mood.
5. Recommendation engine filters matching songs.
6. Top recommended songs are displayed through Streamlit.

---

## 🔮 Future Enhancements

* Live Webcam Emotion Detection
* Spotify API Integration
* Playlist Generation
* User Authentication
* Listening History Tracking
* Hybrid Recommendation System
* Real-Time Mood Monitoring

---

## 👨‍💻 Author

**Prasad Reddy**

GitHub:
https://github.com/prasad2023222

---

## 📜 License

This project is intended for educational and research purposes.
