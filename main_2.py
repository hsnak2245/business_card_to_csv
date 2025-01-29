import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import requests
import json
from PIL import Image
import sys
import subprocess

# Page configuration
st.set_page_config(
    page_title="Business Card Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved error styling
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
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
    }
    
    .stFileUploader {
        background-color: #2d2d2d;
        border-radius: 4px;
        padding: 1rem;
        border: 2px dashed #007bff;
    }
    
    .upload-text {
        color: #007bff;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    .error-box {
        background-color: rgba(220, 53, 69, 0.1);
        color: #dc3545;
        border: 1px solid #dc3545;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        color: #ffc107;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: rgba(0, 123, 255, 0.1);
        color: #007bff;
        border: 1px solid #007bff;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .file-list {
        background-color: #2d2d2d;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .file-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        border-bottom: 1px solid #404040;
    }
    
    .success-box {
        background-color: rgba(40, 167, 69, 0.1);
        color: #28a745;
        border: 1px solid #28a745;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def check_tesseract():
    """Check if tesseract is installed and accessible"""
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True)
        return True
    except FileNotFoundError:
        installation_instructions = """
        Tesseract is not installed or not in your PATH. Please install it:
        
        For Windows:
        1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Add Tesseract to your PATH
        
        For Mac:
        ```brew install tesseract```
        
        For Linux:
        ```sudo apt-get install tesseract-ocr```
        """
        st.error(installation_instructions)
        return False

class BusinessCardProcessor:
    def __init__(self):
        """Initialize the Business Card Processor"""
        self.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
        self.GROQ_API_URL = "https://api.groq.com/v1/chat/completions"
        
    def enhance_image(self, image):
        """Enhance image for better OCR results"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Denoise
            denoised = cv2.fastNlMeansDenoising(gray)
            # Adaptive threshold
            thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            return thresh
        except Exception as e:
            st.error(f"Error enhancing image: {str(e)}")
            return image

    def process_image(self, image_path):
        """Process single image"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                st.error("Failed to load image")
                return None
            
            # Enhance image
            enhanced = self.enhance_image(image)
            
            # Save enhanced image temporarily
            temp_enhanced = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            cv2.imwrite(temp_enhanced.name, enhanced)
            
            # Use tesseract
            import pytesseract
            text = pytesseract.image_to_string(temp_enhanced.name)
            
            # Cleanup
            os.unlink(temp_enhanced.name)
            
            return text.strip()
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None

    def process_video(self, video_path):
        """Process video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Could not open video file")
                return None

            frames_text = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // 10)  # Process 10 frames

            progress_bar = st.progress(0)
            for frame_idx in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Save frame temporarily
                    temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    cv2.imwrite(temp_frame.name, frame)
                    
                    # Process frame
                    text = self.process_image(temp_frame.name)
                    if text:
                        frames_text.append(text)
                    
                    # Cleanup
                    os.unlink(temp_frame.name)
                    
                    # Update progress
                    progress = min(1.0, frame_idx / total_frames)
                    progress_bar.progress(progress)

            cap.release()
            return "\n".join(frames_text)
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return None

def main():
    # Check for tesseract installation
    if not check_tesseract():
        return

    st.title("Business Card Processor")
    st.markdown("Transform your business cards into structured digital data")

    processor = BusinessCardProcessor()

    # File uploader with clear instructions
    st.markdown("""
    <div class="upload-text">
        📤 Upload business card images or videos
        <br>
        <small>Supported formats: JPG, PNG, MP4, AVI</small>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "",  # Empty label since we have custom text above
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Display uploaded files
        st.markdown(f"""
        <div class="info-box">
            📁 {len(uploaded_files)} files ready for processing
        </div>
        """, unsafe_allow_html=True)

        # Process button
        if st.button("Process Business Cards"):
            for uploaded_file in uploaded_files:
                try:
                    # Create temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        file_path = temp_file.name

                    # Process based on file type
                    if uploaded_file.type.startswith('video'):
                        text = processor.process_video(file_path)
                    else:
                        text = processor.process_image(file_path)

                    # Display results
                    if text:
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ Successfully processed: {uploaded_file.name}
                        </div>
                        """, unsafe_allow_html=True)
                        st.text_area(f"Extracted text from {uploaded_file.name}", text, height=150)
                    else:
                        st.markdown(f"""
                        <div class="warning-box">
                            ⚠️ No text could be extracted from {uploaded_file.name}
                        </div>
                        """, unsafe_allow_html=True)

                    # Cleanup
                    os.unlink(file_path)

                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        ❌ Error processing {uploaded_file.name}: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()