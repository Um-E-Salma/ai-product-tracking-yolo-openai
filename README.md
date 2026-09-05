# AI Product Detection & Tracking with YOLO and OpenAI

An intelligent computer vision application that detects, tracks, and identifies products in videos using **YOLOv8, ByteTrack, OpenAI Vision, and OpenCV**.

The system detects objects in each video frame, assigns unique tracking IDs using ByteTrack, and analyzes individual objects using AI to provide product information and estimated prices.

Unlike a basic frame-by-frame detection system, this project uses **object tracking and asynchronous AI processing** to avoid repeatedly analyzing the same object.

---

## Features

* Real-time object detection using YOLOv8
* Object tracking with ByteTrack
* Unique tracking IDs for individual objects
* AI-powered product identification using OpenAI Vision
* Estimated product price generation
* Asynchronous API processing with ThreadPoolExecutor
* Prevents repeated API requests for the same tracked object
* Safe object cropping and image validation
* Image caching for duplicate detections
* Video processing with OpenCV
* Video processing progress tracking
* API timeout and error handling
* Automatic cleanup of inactive object tracks
* Secure environment variable configuration

---

## System Architecture

```text
Input Video
     │
     ▼
Frame Extraction
     │
     ▼
YOLOv8 Object Detection
     │
     ▼
ByteTrack Object Tracking
     │
     ▼
Unique Track ID
     │
     ├──────────────────────┐
     ▼                      │
Object Crop                 │
     │                      │
     ▼                      │
OpenAI Vision API           │
     │                      │
     ▼                      │
Product Identification      │
     │                      │
     ▼                      │
Product Name + Estimated Price
     │
     ▼
Async Result Processing
     │
     ▼
Bounding Box + AI Label
     │
     ▼
Processed Output Video
```

---

## Technologies Used

* Python
* YOLOv8
* Ultralytics
* ByteTrack
* OpenAI Vision API
* OpenCV
* ThreadPoolExecutor
* Computer Vision
* Object Detection
* Object Tracking
* Artificial Intelligence

---

## Project Structure

```text
ai-product-tracking-yolo-openai/
│
├── app.py
├── video_processor.py
│
├── uploads/
│   ├── input_video.mp4
│   └── processed_output_video.mp4
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Um-E-Salma/ai-product-tracking-yolo-openai.git
cd ai-product-tracking-yolo-openai
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory of the project.

```env
OPENAI_KEY=your_openai_api_key_here
```

You can use the provided `.env.example` file as a reference:

```env
OPENAI_KEY=your_openai_api_key_here
```

---

## Usage

Place your input video inside the `uploads` directory.

Example:

```text
uploads/input_video.mp4
```

Run the application:

```bash
python video_processor.py
```

The application will:

1. Load the YOLOv8 model.
2. Read the input video frame by frame.
3. Detect objects using YOLOv8.
4. Track objects using ByteTrack.
5. Assign a unique tracking ID to each object.
6. Wait until the object is consistently detected.
7. Send the object image to OpenAI Vision for product identification.
8. Receive product information and an estimated price.
9. Draw the result on the video.
10. Save the processed output video.

---

## How Object Tracking Works

Instead of sending every detected object from every frame to an AI API, the system uses **ByteTrack** to maintain a unique identity for each object.

Example:

```text
Frame 1:

Phone → Track ID: 7

Frame 2:

Same Phone → Track ID: 7

Frame 3:

Same Phone → Track ID: 7
```

The system recognizes that this is the same object and avoids unnecessary repeated AI requests.

---

## Asynchronous AI Processing

OpenAI API requests are executed in background threads using:

```python
ThreadPoolExecutor
```

This allows the video processing pipeline to continue while product information is being generated.

Each tracked object maintains its own processing state:

```text
New Object
    ↓
Tracking
    ↓
Stable Detection
    ↓
AI Request Submitted
    ↓
Processing
    ↓
Product Information Received
    ↓
Cached Result
```

---

## Error Handling

The application handles:

* Missing OpenAI API keys
* Video loading errors
* YOLO model loading failures
* Invalid bounding boxes
* Empty image crops
* API request timeouts
* API request failures
* Object tracking failures
* Video writer errors

---

## Example Output

The processed video displays information similar to:

```text
ID 7 | Product Name: iPhone, Estimated Price: $799

ID 12 | Product Name: Coca-Cola Bottle, Estimated Price: $2

ID 18 | Product Name: Laptop, Estimated Price: Unknown
```

---

## Future Improvements

* [ ] FastAPI REST API integration
* [ ] Web-based video upload interface
* [ ] Database integration for detected products
* [ ] Product price comparison using retailer APIs
* [ ] Object detection analytics dashboard
* [ ] Docker containerization
* [ ] GPU inference optimization
* [ ] Redis caching
* [ ] Celery background task processing
* [ ] WebSocket-based real-time progress updates

---

## Use Cases

This project can be adapted for:

* Smart retail systems
* Inventory monitoring
* Product recognition
* AI-powered video analytics
* Retail automation
* Store analytics
* Visual product search
* Intelligent computer vision applications

---

## Author

**Um-E-Salma**

AI Engineer | Machine Learning Engineer | Computer Vision | Generative AI | MLOps

---

## ⭐ Support

If you find this project useful, consider giving it a star on GitHub!
