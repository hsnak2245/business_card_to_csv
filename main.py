import streamlit as st
import cv2
import pytesseract
import tempfile
import csv
import os
from groq import Groq
from PIL import Image
import numpy as np
from typing import List, Dict
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up page configuration
st.set_page_config(
    page_title="Business Card Processor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling with modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main {
        background-color: #1E1E1E;
        color: #E0E0E0;
        font-family: 'Montserrat', sans-serif;
        padding: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2D2D2D;
    }
    
    /* Headers */
    h1 {
        color: #FF0000 !important;
        font-weight: 600;
        margin-bottom: 2rem;
    }
    
    h2, h3 {
        color: #E0E0E0 !important;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #FF0000;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-family: 'Montserrat', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #CC0000;
        transform: translateY(-2px);
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #2D2D2D;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #FF0000;
    }
    
    /* Tables */
    .dataframe {
        background-color: #2D2D2D;
        color: #E0E0E0;
        border-radius: 4px;
    }
    
    /* Multiselect */
    .stMultiSelect {
        background-color: #2D2D2D;
        color: #E0E0E0;
    }
    
    /* Error messages */
    .stAlert {
        background-color: #FF000020;
        color: #FF0000;
        border: 1px solid #FF0000;
        border-radius: 4px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

class BusinessCardProcessor:
    def __init__(self):
        """Initialize the Business Card Processor with configuration"""
        self.groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.available_columns = [
            "Name", "Company", "Position", "Phone", 
            "Email", "Website", "Address", "Other"
        ]
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.video_extensions = ['.mp4', '.avi', '.mov']

    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised

    def extract_frames(self, video_path: str, interval: int = 30) -> List[np.ndarray]:
        """Extract frames from video at specified intervals with enhanced error handling"""
        frames = []
        try:
            # Ensure video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError("Video file not found")

            # Open video file
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                raise ValueError("Unable to open video file")

            # Get video properties
            total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("Invalid video file or no frames detected")

            fps = vidcap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            # Log video information
            logger.info(f"Video properties - Frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")

            # Calculate frame extraction interval
            frame_interval = max(1, int(fps * interval / 30))  # Adjust interval based on video FPS
            
            with st.progress(0) as progress_bar:
                frame_count = 0
                while True:
                    # Read frame
                    success, image = vidcap.read()
                    if not success:
                        break

                    # Process frame at intervals
                    if frame_count % frame_interval == 0:
                        # Verify frame integrity
                        if image is not None and image.size > 0:
                            # Check if frame is not too dark or too bright
                            if np.mean(image) > 10 and np.mean(image) < 245:
                                frames.append(image)
                        
                    # Update progress
                    if frame_count % 10 == 0:  # Update progress every 10 frames
                        progress = min(1.0, frame_count / total_frames)
                        progress_bar.progress(progress)

                    frame_count += 1

            # Cleanup
            vidcap.release()

            # Verify we extracted some frames
            if not frames:
                raise ValueError("No valid frames extracted from video")

            logger.info(f"Successfully extracted {len(frames)} frames from video")
            return frames

        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            st.error("Video file not found. Please check the file and try again.")
            return []
        except ValueError as e:
            logger.error(f"Video processing error: {str(e)}")
            st.error(f"Error processing video: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in video processing: {str(e)}")
            st.error("Unexpected error while processing video. Please try a different file or format.")
            return []
        finally:
            # Ensure video capture is released
            if 'vidcap' in locals():
                vidcap.release()

    def process_image(self, image: np.ndarray) -> str:
        """Process image with enhanced OCR"""
        try:
            # Enhance image
            enhanced_image = self.enhance_image(image)
            
            # Configure OCR settings
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR processing error: {str(e)}")
            return ""

    def process_with_groq(self, text: str) -> str:
        """Process extracted text with Groq LLM using improved prompt"""
        try:
            prompt = f"""Extract and structure the following business card information:
            {text}
            
            Rules:
            1. Maintain consistent formatting
            2. Validate email and phone formats
            3. Clean and standardize addresses
            4. Use 'N/A' for missing fields
            
            Return only in CSV format with header:
            Name,Company,Position,Phone,Email,Website,Address,Other"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq processing error: {str(e)}")
            return ""

    def process_files(self, uploaded_files: List) -> List[Dict]:
        """Process uploaded files and return structured data"""
        all_data = []
        
        for uploaded_file in uploaded_files:
            try:
                # Validate file size
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
                if file_size > 100:  # 100MB limit
                    st.warning(f"File {uploaded_file.name} is too large ({file_size:.1f}MB). Skipping...")
                    continue

                # Create progress bar for each file
                progress_text = f"Processing {uploaded_file.name}..."
                with st.spinner(progress_text):
                    # Create temp directory if it doesn't exist
                    temp_dir = tempfile.mkdtemp()
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    # Save file with original name
                    with open(file_path, 'wb') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())

                    # Process based on file type
                    if any(uploaded_file.name.lower().endswith(ext) for ext in self.video_extensions):
                        frames = self.extract_frames(file_path)
                        for frame in frames:
                            text = self.process_image(frame)
                            if text:
                                csv_data = self.process_with_groq(text)
                                if csv_data:
                                    all_data.extend(csv_data.split('\n')[1:])
                    
                    elif any(uploaded_file.name.lower().endswith(ext) for ext in self.image_extensions):
                        image = cv2.imread(file_path)
                        text = self.process_image(image)
                        if text:
                            csv_data = self.process_with_groq(text)
                            if csv_data:
                                all_data.extend(csv_data.split('\n')[1:])
                    
                    # Cleanup
                    os.unlink(file_path)
                    
            except Exception as e:
                logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                st.error(f"Error processing {uploaded_file.name}. Skipping to next file.")
                continue
        
        return all_data

def main():
    processor = BusinessCardProcessor()
    
    # App header
    st.title("📇 Business Card Processor")
    st.markdown("""
    Transform your business cards into structured digital data.
    Upload images or videos of business cards to extract information.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Column selection
        selected_columns = st.multiselect(
            "Select columns to include",
            processor.available_columns,
            default=processor.available_columns
        )
        
        # Advanced settings
        st.subheader("Advanced Settings")
        video_interval = st.slider(
            "Video frame interval",
            min_value=15,
            max_value=60,
            value=30,
            help="Number of frames to skip in video processing"
        )

    # Main content area
    uploaded_files = st.file_uploader(
        "Upload Business Cards",
        type=[ext[1:] for ext in processor.image_extensions + processor.video_extensions],
        accept_multiple_files=True
    )

    # Initialize session state for processed data if it doesn't exist
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if uploaded_files:
        # Add process button
        if st.button("🔄 Process Business Cards", type="primary", use_container_width=True):
            start_time = time.time()
            
            # Process files
            with st.spinner("Processing business cards... Please wait."):
                all_data = processor.process_files(uploaded_files)
                st.session_state.processed_data = all_data
        
        # Show number of files uploaded
        st.info(f"📁 {len(uploaded_files)} files uploaded and ready for processing")
        
        # Display results
        if st.session_state.processed_data:
            # Create filtered data
            reader = csv.DictReader(st.session_state.processed_data, fieldnames=processor.available_columns)
            filtered_data = [{col: row[col] for col in selected_columns} for row in reader]
            
            # Display processing summary
            st.success(f"""
            ✅ Processing Complete!
            - Processed {len(uploaded_files)} files
            - Extracted {len(filtered_data)} business cards
            - Processing time: {time.time() - start_time:.2f} seconds
            """)
            
            # Show data table
            st.subheader("📊 Extracted Data")
            st.dataframe(filtered_data, use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Options")
            col1, col2 = st.columns(2)
            
            # CSV export
            with col1:
                csv_string = "\n".join([",".join(row.values()) for row in filtered_data])
                st.download_button(
                    label="Download CSV",
                    data=csv_string,
                    file_name="business_cards.csv",
                    mime="text/csv",
                    key="csv_download"
                )
            
            # JSON export
            with col2:
                import json
                json_string = json.dumps(filtered_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_string,
                    file_name="business_cards.json",
                    mime="application/json",
                    key="json_download"
                )

if __name__ == "__main__":
    main()