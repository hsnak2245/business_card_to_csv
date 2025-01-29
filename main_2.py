import streamlit as st
import cv2
import pytesseract
import numpy as np
import tempfile
import os
import requests
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Business Card Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600&display=swap');
    
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
    }
    
    h1 {
        color: #007bff !important;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 500;
        width: 100%;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
    }
    
    .stFileUploader {
        background-color: #2d2d2d;
        border-radius: 4px;
        padding: 1rem;
    }
    
    .stAlert {
        background-color: rgba(0, 123, 255, 0.1);
        color: #007bff;
        border: 1px solid #007bff;
        border-radius: 4px;
        padding: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"

class BusinessCardProcessor:
    @staticmethod
    def enhance_image(frame):
        """Enhance image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        return thresh

    @staticmethod
    def extract_text_from_frame(frame):
        """Extract text from a single frame"""
        try:
            # Enhance image
            processed_frame = BusinessCardProcessor.enhance_image(frame)
            # Extract text using pytesseract
            text = pytesseract.image_to_string(processed_frame)
            return text.strip()
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return ""

    @staticmethod
    def process_video(video_path):
        """Process video file and extract text from frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                return ""

            text_output = ""
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_size = max(1, frame_count // 10)  # Extract text from 10 keyframes

            with st.progress(0) as progress_bar:
                for i in range(0, frame_count, step_size):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        text_output += BusinessCardProcessor.extract_text_from_frame(frame) + "\n"
                    progress_bar.progress(min(1.0, (i + 1) / frame_count))

            cap.release()
            return text_output.strip()
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return ""

    @staticmethod
    def analyze_text_with_groq(text):
        """Analyze extracted text using Groq API"""
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Extract business card information from the following text:
        {text}
        
        Format the response as structured data with:
        - Name
        - Company
        - Position
        - Phone
        - Email
        - Website
        - Address
        """
        
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }

        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")
            return "API request failed."

def main():
    st.title("Business Card Processor")
    st.markdown("Transform your business cards into structured digital data. Upload images or videos to extract information.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Business Cards",
        type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files uploaded and ready for processing")

        if st.button("Process Business Cards"):
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        file_path = temp_file.name

                    try:
                        # Process based on file type
                        if uploaded_file.type.startswith('video'):
                            extracted_text = BusinessCardProcessor.process_video(file_path)
                        else:  # Image file
                            image = cv2.imread(file_path)
                            extracted_text = BusinessCardProcessor.extract_text_from_frame(image)

                        if extracted_text:
                            # Analyze with Groq
                            analysis_result = BusinessCardProcessor.analyze_text_with_groq(extracted_text)
                            
                            # Display results in expandable section
                            with st.expander(f"Results for {uploaded_file.name}"):
                                st.text_area("Extracted Text", extracted_text, height=100)
                                st.text_area("Structured Information", analysis_result, height=200)
                        else:
                            st.warning(f"No text could be extracted from {uploaded_file.name}")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Cleanup
                        if os.path.exists(file_path):
                            os.remove(file_path)

if __name__ == "__main__":
    main()