import cv2
import numpy as np
import pytesseract
import spacy
import re
import csv
import threading
import time
from datetime import datetime
from urllib.parse import urlparse
import logging
from queue import Queue
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BusinessCardScanner:
    def __init__(self, ip_cam_url=None, csv_path='business_cards.csv'):
        self.ip_cam_url = ip_cam_url or 'http://192.168.1.100:8080/video'
        self.csv_path = csv_path
        self.frame_queue = Queue(maxsize=30)
        self.stop_event = threading.Event()
        
        # Colors for visualization
        self.colors = {
            'Name': (0, 255, 0),      # Green
            'Position': (255, 0, 0),   # Blue
            'Company': (0, 0, 255),    # Red
            'Phone': (255, 255, 0),    # Cyan
            'Email': (255, 0, 255),    # Magenta
            'Website': (0, 255, 255)   # Yellow
        }
        
        # Initialize NLP
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded SpaCy NLP model")
        except OSError:
            logger.error("Failed to load SpaCy model. Installing...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')

        # Initialize CSV file with headers
        self._init_csv()
        
        # OCR Configuration
        self.custom_config = r'--oem 3 --psm 11'
        
        # Motion detection parameters
        self.motion_threshold = 1000
        self.stable_frames_required = 30
        self.stable_frame_count = 0
        self.prev_frame = None

    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        headers = ['Name', 'Position', 'Company', 'Phone', 'Email', 'Website', 'Timestamp']
        try:
            with open(self.csv_path, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                logger.info(f"Created new CSV file: {self.csv_path}")
        except FileExistsError:
            logger.info(f"Using existing CSV file: {self.csv_path}")

    def _camera_stream(self):
        """Thread function to capture camera stream"""
        cap = cv2.VideoCapture(self.ip_cam_url)
        if not cap.isOpened():
            logger.error("Failed to connect to IP camera")
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                continue

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        cap.release()

    def _preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Reduce noise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        return denoised

    def _detect_motion(self, frame):
        """Detect if business card is stable in frame"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        motion = np.sum(thresh) > self.motion_threshold
        
        if not motion:
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 0
            
        self.prev_frame = gray
        return self.stable_frame_count >= self.stable_frames_required

    def _extract_text_with_boxes(self, image):
        """Extract text from image using Tesseract OCR with bounding boxes"""
        try:
            preprocessed = self._preprocess_image(image)
            
            # Get text and bounding boxes
            text = pytesseract.image_to_string(preprocessed, config=self.custom_config)
            boxes = pytesseract.image_to_data(preprocessed, config=self.custom_config, 
                                            output_type=pytesseract.Output.DICT)
            
            # Create a mapping of text locations
            text_locations = []
            n_boxes = len(boxes['text'])
            for i in range(n_boxes):
                if int(boxes['conf'][i]) > 60:  # Confidence threshold
                    text_locations.append({
                        'text': boxes['text'][i],
                        'x': boxes['left'][i],
                        'y': boxes['top'][i],
                        'w': boxes['width'][i],
                        'h': boxes['height'][i]
                    })
            
            return text.strip(), text_locations
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return "", []

    def _parse_business_card(self, text, image):
        """Parse extracted text into structured data and annotate image"""
        original_image = image.copy()
        # Initialize result dictionary
        result = {
            'Name': '',
            'Position': '',
            'Company': '',
            'Phone': '',
            'Email': '',
            'Website': ''
        }
        
        # Process text with SpaCy
        doc = self.nlp(text)
        
        # Extract name (assuming first PERSON entity is the name)
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and not result['Name']:
                result['Name'] = ent.text
                break
        
        # Extract email
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if emails:
            result['Email'] = emails[0]
        
        # Extract phone numbers
        phones = re.findall(
            r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            text
        )
        if phones:
            result['Phone'] = phones[0]
        
        # Extract website
        urls = re.findall(r'(?:www\.)?[\w\.-]+\.\w+', text)
        if urls:
            result['Website'] = urls[0]
        
        # Extract position (looking for common job title patterns)
        job_titles = [
            'CEO', 'CTO', 'CFO', 'Director', 'Manager', 'Engineer',
            'Developer', 'Analyst', 'Consultant', 'President',
            'Vice President', 'VP', 'Associate', 'Assistant'
        ]
        for line in text.split('\n'):
            for title in job_titles:
                if title.lower() in line.lower():
                    result['Position'] = line.strip()
                    break
            if result['Position']:
                break
        
        # Extract company name (assuming it's often on a line by itself)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (len(line) > 2 and
                not any(key in line.lower() for key in 
                    ['@', 'www', '.com', 'phone', 'tel', 'email']) and
                not any(char.isdigit() for char in line) and
                line not in [result['Name'], result['Position']]):
                result['Company'] = line
                break
        
        return result

    def _is_duplicate(self, new_data, existing_data):
        """Check if the new entry is a duplicate using fuzzy matching"""
        # Function to normalize text for comparison
        def normalize(text):
            return re.sub(r'\W+', '', text.lower()) if text else ''
        
        # Check exact matches for email and phone
        if (new_data['Email'] and new_data['Email'] == existing_data['Email']) or \
           (new_data['Phone'] and new_data['Phone'] == existing_data['Phone']):
            return True
            
        # Check name similarity (allowing for slight variations)
        name_similarity = len(set(normalize(new_data['Name'])) & 
                            set(normalize(existing_data['Name']))) / \
                        max(len(normalize(new_data['Name'])), 
                            len(normalize(existing_data['Name'])), 1)
                            
        # Check company name similarity
        company_similarity = len(set(normalize(new_data['Company'])) & 
                               set(normalize(existing_data['Company']))) / \
                           max(len(normalize(new_data['Company'])), 
                               len(normalize(existing_data['Company'])), 1)
        
        # Consider it a duplicate if both name and company are very similar
        if name_similarity > 0.8 and company_similarity > 0.8:
            return True
            
        return False

    def _save_to_csv(self, data):
        """Save extracted data to CSV file"""
        data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Check for duplicates
        try:
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._is_duplicate(data, row):
                        logger.info("Duplicate entry found, skipping...")
                        return False
        except FileNotFoundError:
            pass  # File doesn't exist yet
        
        # Append new entry
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writerow(data)
            logger.info("Saved new business card entry")
        return True

    def start(self):
        """Start the business card scanning process"""
        logger.info("Starting business card scanner...")
        
        # Start camera stream thread
        stream_thread = threading.Thread(target=self._camera_stream)
        stream_thread.start()
        
        try:
            while not self.stop_event.is_set():
                if self.frame_queue.empty():
                    continue
                    
                frame = self.frame_queue.get()
                
                # Show live preview
                cv2.imshow('Business Card Scanner', frame)
                
                # Check for stable card
                if self._detect_motion(frame):
                    logger.info("Stable card detected, capturing...")
                    
                    # Extract and process text with locations
                    text, text_locations = self._extract_text_with_boxes(frame)
                    if text:
                        data = self._parse_business_card(text, frame)
                        if any(data.values()):
                            # Create visualization
                            vis_frame = frame.copy()
                            
                            # Draw bounding boxes and labels for detected information
                            for field, value in data.items():
                                if value and field in self.colors:
                                    # Find matching text location
                                    for loc in text_locations:
                                        if value.lower() in loc['text'].lower():
                                            x, y, w, h = loc['x'], loc['y'], loc['w'], loc['h']
                                            # Draw rectangle
                                            cv2.rectangle(vis_frame, 
                                                        (x, y), 
                                                        (x + w, y + h), 
                                                        self.colors[field], 2)
                                            # Draw label
                                            cv2.putText(vis_frame, 
                                                       field,
                                                       (x, y - 10),
                                                       cv2.FONT_HERSHEY_SIMPLEX,
                                                       0.5,
                                                       self.colors[field],
                                                       1)
                            
                            if self._save_to_csv(data):
                                logger.info("Successfully processed business card")
                                # Visual feedback
                                cv2.putText(
                                    vis_frame,
                                    "Card Captured!",
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 0),
                                    2
                                )
                            else:
                                cv2.putText(
                                    vis_frame,
                                    "Duplicate Card!",
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    2
                                )
                            
                            cv2.imshow('Business Card Scanner', vis_frame)
                            cv2.waitKey(1000)
                    
                    # Reset motion detection
                    self.stable_frame_count = 0
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Stopping scanner...")
        finally:
            self.stop_event.set()
            stream_thread.join()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    scanner = BusinessCardScanner()
    scanner.start()