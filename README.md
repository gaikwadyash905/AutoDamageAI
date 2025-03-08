# AutoDamageAI

An AI-powered computer vision system for automated vehicle damage detection and assessment

## Features

- Upload car images to detect damages
- AI-based detection of scratches, dents, broken glass, broken lights, and major damage
- Visual highlighting of damage areas in the image with color-coded bounding boxes
- Damage severity assessment and confidence levels
- Real-time processing with detailed analysis
- Responsive design that works on desktop and mobile devices

## Technologies Used

- **Backend**: Python, Flask 2.3.3
- **AI/ML**: YOLOv8, OpenCV, PyTorch 2.1.0
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Image Processing**: OpenCV-Python-Headless 4.8.0.76, Pillow 10.0.0

## Installation & Setup

1. Clone this repository to your local machine.


2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://127.0.0.1:5000/`

## Usage

1. Open the application in your web browser
2. Upload an image of a car with visible damage
3. The system will automatically:
   - Process the image using YOLOv8
   - Detect and classify different types of damage
   - Draw color-coded bounding boxes around detected damages
   - Show confidence levels for each detection
   - Provide an overall damage severity assessment

## How It Works

1. **Upload** → User uploads car image through web interface
2. **Process** → AI model (YOLOv8 + OpenCV) analyzes the image
3. **Detect** → System identifies damages (scratches, dents, glass, lights)
4. **Analyze** → Calculates severity and confidence levels
5. **Display** → Shows results with color-coded damage markers

## Damage Types and Color Codes

- Scratches: Cyan
- Dents: Orange
- Broken Glass: Red
- Broken Lights: Blue
- Major Damage: Red

## Technical Details

- Maximum file upload size: 16MB
- Supported image formats: PNG, JPG, JPEG
- Image validation checks:
  - File format verification
  - Minimum dimensions: 10x10
  - Maximum dimensions: 10000x10000
  - File integrity check

## Limitations

- The default YOLOv8n model is not specifically trained for car damage detection. For optimal results, a custom-trained model on car damage data should be used.
- Detection accuracy depends on image quality, lighting, and angle.
- The system may generate false positives or miss some damages in complex scenarios.

## Error Handling

- Invalid image uploads are detected and reported
- Processing errors are gracefully handled with user feedback
- Automatic cleanup of temporary files
- Session-based file management for security

## Component Details

### Frontend Components
- **User Interface**: HTML templates with Bootstrap 5 for responsive design
- **JavaScript**: Handles file uploads, form submissions, and dynamic updates
- **CSS**: Custom styling and Bootstrap customization

### Backend Components
- **Flask Application**: Main server handling routes and requests
- **Session Management**: Tracks user sessions and manages file associations
- **File Controller**: Handles file uploads, downloads, and cleanup
- **Image Management**: Validates and processes uploaded images

### AI Processing Pipeline
- **YOLOv8 Model**: Pre-trained model for object detection
- **Image Processing**: OpenCV-based image manipulation and analysis
- **Deep Learning**: PyTorch backend for model inference
- **Damage Analysis**: Classification and severity assessment

### Storage System
- **Uploads Folder**: Temporary storage for user-uploaded images
- **Results Folder**: Stores processed images with damage annotations
- **Model Storage**: Contains YOLOv8 model weights

### Error Handling & Validation
- **Image Validation**: Checks file format, size, and dimensions
- **Error Handlers**: Custom error pages and API error responses
- **Cleanup Service**: Automatic removal of temporary files

## Data Flow

1. User uploads an image through the web interface
2. Frontend JavaScript handles the file and sends it to the backend
3. Backend validates the image and stores it temporarily
4. Image is processed through the AI pipeline:
   - YOLOv8 model detects damages
   - OpenCV draws annotations
   - Results are analyzed for severity
5. Processed image and analysis are stored
6. Results are sent back to frontend for display
7. Cleanup service removes temporary files after processing
