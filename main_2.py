import streamlit as st
import cv2
import pytesseract
import numpy as np
import tempfile
import os
import requests
import json

# Groq API Configuration
GROQ_API_KEY = "your_groq_api_key"
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"

def extract_text_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return ""
    
    text_output = ""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_size = max(1, frame_count // 10)  # Extract text from 10 keyframes
    
    for i in range(0, frame_count, step_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        text_output += extract_text_from_frame(frame) + "\n"
    
    cap.release()
    return text_output.strip()

def analyze_text_with_groq(text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": text}]
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response from API.")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return "API request failed."

st.title("Video OCR & AI Analysis")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
    
    st.video(video_path)
    st.write("Extracting text from video...")
    extracted_text = process_video(video_path)
    
    if extracted_text:
        st.text_area("Extracted Text", extracted_text, height=150)
        st.write("Analyzing extracted text with Groq AI...")
        analysis_result = analyze_text_with_groq(extracted_text)
        st.text_area("AI Analysis", analysis_result, height=200)
    else:
        st.warning("No text could be extracted from the video.")
    
    os.remove(video_path)
