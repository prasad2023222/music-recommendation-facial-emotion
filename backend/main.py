from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image
import pandas as pd
import numpy as np
import io

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Emotion Music Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD EMOTION MODEL
# -----------------------------

class_names = {
    0: "sad",
    1: "angry",
    2: "neutral",
    3: "happy",
    4: "fear",
    5: "disgust",
    6: "surprise"
}

emotion_model = models.resnet18(weights=None)

emotion_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(emotion_model.fc.in_features, 7)
)

emotion_model.load_state_dict(
    torch.load(
        "best_resnet18_emotion_model.pth",
        map_location=device
    )
)

emotion_model.to(device)
emotion_model.eval()

emotion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD SONG DATASET
# -----------------------------

df = pd.read_csv("data_moods.csv")

features = [
    'popularity',
    'length',
    'danceability',
    'acousticness',
    'energy',
    'instrumentalness',
    'liveness',
    'valence',
    'loudness',
    'speechiness',
    'tempo',
    'key',
    'time_signature'
]

df = df.dropna(subset=features + ["mood"])

scaler = StandardScaler()
song_features = scaler.fit_transform(df[features])

emotion_to_mood = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Calm",
    "fear": "Calm",
    "disgust": "Calm",
    "neutral": "Calm",
    "surprise": "Energetic"
}

# -----------------------------
# PREDICT EMOTION
# -----------------------------

def predict_emotion(image):

    image_tensor = emotion_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = emotion_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    emotion = class_names[predicted.item()]

    return emotion, float(confidence.item() * 100)

# -----------------------------
# RECOMMEND SONGS
# -----------------------------

def recommend_songs(emotion, top_n=10):

    mood = emotion_to_mood[emotion]

    mood_songs = df[
        df["mood"] == mood
    ].copy()

    if len(mood_songs) == 0:
        return mood, []

    mood_indices = mood_songs.index.tolist()

    mood_vectors = song_features[
        mood_indices
    ]

    user_vector = np.mean(
        mood_vectors,
        axis=0
    ).reshape(1, -1)

    similarities = cosine_similarity(
        user_vector,
        mood_vectors
    )[0]

    mood_songs["similarity"] = similarities

    recommendations = mood_songs.sort_values(
        by=["similarity", "popularity"],
        ascending=False
    ).head(top_n)

    result = recommendations[
        [
            "name",
            "artist",
            "album",
            "popularity",
            "mood"
        ]
    ]

    return mood, result.to_dict(
        orient="records"
    )

# -----------------------------
# API ENDPOINT
# -----------------------------

@app.get("/")
def home():
    return {
        "message":
        "Emotion Music Recommendation API Running"
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    emotion, confidence = predict_emotion(
        image
    )

    music_mood, songs = recommend_songs(
        emotion
    )

    return {
        "emotion": emotion,
        "confidence": confidence,
        "music_mood": music_mood,
        "songs": songs
    }