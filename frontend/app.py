import streamlit as st
import requests
from PIL import Image
import pandas as pd

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Emotion Based Music Recommender",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Recommendation Based on Facial Emotion Recognition")

st.write("Upload your face image. The system detects your emotion and recommends songs.")

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Emotion & Recommend Songs"):
        with st.spinner("Analyzing emotion and recommending music..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()

                    emotion = data["emotion"]
                    confidence = data["confidence"]
                    music_mood = data["music_mood"]
                    songs = data["songs"]

                    with col2:
                        st.success("Prediction Completed")

                        st.subheader("Detected Emotion")
                        st.write(f"**Emotion:** {emotion}")
                        st.write(f"**Confidence:** {confidence:.2f}%")
                        st.write(f"**Recommended Music Mood:** {music_mood}")

                    st.subheader("Recommended Songs")

                    if len(songs) > 0:
                        songs_df = pd.DataFrame(songs)
                        st.dataframe(songs_df, use_container_width=True)

                        for i, song in enumerate(songs, start=1):
                            st.markdown(
                                f"""
                                **{i}. {song['name']}**  
                                Artist: {song['artist']}  
                                Album: {song['album']}  
                                Popularity: {song['popularity']}  
                                Mood: {song['mood']}
                                ---
                                """
                            )
                    else:
                        st.warning("No songs found for this mood.")

                else:
                    st.error("FastAPI backend error")
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error("FastAPI backend is not running.")
                st.write("Start backend first using:")
                st.code("uvicorn main:app --reload")