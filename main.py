import streamlit as st
import cv2
import pytesseract
import tempfile
import csv
import os
from groq import Groq
from PIL import Image

# Set up page configuration
st.set_page_config(
    page_title="BizCard Processor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    body {
        background-color: #FFFFFF;
        color: #333333;
        font-family: 'Montserrat', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
    }
    h1, h2, h3 {
        color: #FF0000 !important;
    }
    .stButton>button {
        background-color: #FF0000;
        color: white;
        border-radius: 4px;
    }
    .stFileUploader>div>div>div>div {
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def extract_frames(video_path, interval=30):
    """Extract frames from video at specified intervals"""
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    success = True
    
    while success:
        success, image = vidcap.read()
        if count % interval == 0 and success:
            frames.append(image)
        count += 1
    return frames

def process_image(image):
    """Process image with OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def process_with_groq(text):
    """Process extracted text with Groq LLM"""
    prompt = f"""Process this business card information into structured data:
    {text}
    
    Return only CSV format with columns: Name,Company,Position,Phone,Email,Website,Address,Other. 
    Use 'N/A' for missing information."""
    
    response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="mixtral-8x7b-32768",
        temperature=0.1
    )
    return response.choices[0].message.content

def main():
    st.title("Business Card Processor")
    st.markdown("Upload video/images of business cards to extract information")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Video or Images",
        type=["mp4", "avi", "mov", "jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    # Column selection
    available_columns = ["Name", "Company", "Position", "Phone", "Email", "Website", "Address", "Other"]
    selected_columns = st.multiselect("Select columns to include in CSV", available_columns, default=available_columns)

    if uploaded_files:
        all_data = []
        
        for uploaded_file in uploaded_files:
            # Temporary file storage
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            # Process based on file type
            if uploaded_file.type.startswith('video'):
                frames = extract_frames(file_path)
                for frame in frames:
                    text = process_image(frame)
                    if text.strip():
                        csv_data = process_with_groq(text)
                        all_data.extend(csv_data.split('\n')[1:])  # Skip header
            else:
                image = cv2.imread(file_path)
                text = process_image(image)
                if text.strip():
                    csv_data = process_with_groq(text)
                    all_data.extend(csv_data.split('\n')[1:])

            os.unlink(file_path)

        # Process and display results
        if all_data:
            # Create CSV
            reader = csv.DictReader(all_data, fieldnames=available_columns)
            filtered_data = [{col: row[col] for col in selected_columns} for row in reader]

            # Show data
            st.subheader("Processed Data")
            st.table(filtered_data[:5])  # Show first 5 entries

            # Download button
            csv_string = "\n".join([",".join(row.values()) for row in filtered_data])
            st.download_button(
                label="Download CSV",
                data=csv_string,
                file_name="business_cards.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
