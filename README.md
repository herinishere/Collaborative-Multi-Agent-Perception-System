# 🧠 Real-Time AI Perception System

A multi-agent computer vision pipeline that combines **object detection**, **image captioning**, **LLM narration**, and **temporal memory** to describe what's happening in real time — from a webcam, image, or video file.

---

## 📌 What It Does

| Agent | Role |
|---|---|
| **VisionAgent** | Detects objects using YOLOv8 |
| **ContextAgent** | Generates a scene caption using BLIP |
| **LanguageAgent** | Narrates the scene in one sentence via Groq LLM |
| **CriticAgent** | Validates the narration for consistency |
| **MemoryAgent** | Tracks object/scene changes over time |

The agents are wired together using **LangGraph**, forming a self-correcting pipeline that retries if the LLM's narration contradicts the detected objects.

---

## 🖥️ System Requirements

- Python 3.9 or higher
- A working webcam (for webcam mode)
- Internet connection (for Groq API calls)
- At least 4 GB RAM recommended
- GPU optional but speeds up YOLO + BLIP inference

---

## 📦 Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-repo/perception-system.git
cd perception-system
```

Or just place the `main.py` file in a folder of your choice.

---

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install opencv-python pillow numpy ultralytics transformers groq langgraph
```

Full list of packages installed:

| Package | Purpose |
|---|---|
| `opencv-python` | Video capture and display |
| `pillow` | Image conversion for BLIP |
| `numpy` | Frame array handling |
| `ultralytics` | YOLOv8 object detection |
| `transformers` | BLIP image captioning model |
| `groq` | LLM API client (Llama 3.1) |
| `langgraph` | Agent workflow graph |

---

### 4. Get a Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** → **Create API Key**
4. Copy your key

---

### 5. Add your API key to the code

Open `main.py` and find this line near the top:

```python
client = Groq(api_key="YOUR_GROQ_API_KEY")
```

Replace `YOUR_GROQ_API_KEY` with your actual key:

```python
client = Groq(api_key="gsk_xxxxxxxxxxxxxxxxxxxx")
```

Alternatively, use an environment variable for security:

```python
import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
```

Then set it in your terminal:

```bash
# macOS/Linux
export GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxx"

# Windows
set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Running the System

```bash
python main.py
```

You will see:

```
1→Webcam | 2→Image | 3→Video
Choice:
```

---

## 📷 Usage Modes

### Mode 1 — Webcam (Live)

```
Choice: 1
```

- Opens your default webcam (device index `0`)
- Runs detection + narration every 30 frames
- Updates memory summary every 1.5 seconds
- Press **`q`** to quit

---

### Mode 2 — Single Image

```
Choice: 2
Image path: /path/to/your/image.jpg
```

- Runs the full pipeline once on the image
- Displays the result in a window
- Press **any key** to close

**Tips for image path:**
- Use full absolute paths to avoid errors
- On Windows, use forward slashes: `C:/Users/YourName/Pictures/photo.jpg`
- Spaces in the path are fine — quotes are stripped automatically

---

### Mode 3 — Video File

```
Choice: 3
Video Path: /path/to/your/video.mp4
```

- Processes the video frame by frame
- Runs the agent pipeline every 30 frames
- Press **`q`** to quit

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv` — anything OpenCV can open.

---

## 🪟 On-Screen Overlay Explained

When the system is running, you'll see an overlay on the video frame:

```
FPS: 24.3
CONTEXT: A person sitting at a desk
LLM: A person is working at their computer in an office.
CRITIC: Accurate.
MEM: Persistent: chair, keyboard, person | Just appeared: cup
DET: CHAIR x1 | KEYBOARD x1 | PERSON x1
```

| Label | Description |
|---|---|
| **FPS** | Frames per second being rendered |
| **CONTEXT** | Raw BLIP image caption |
| **LLM** | One-sentence narration from Groq |
| **CRITIC** | Consistency check result (green = OK, red = issue) |
| **MEM** | Temporal memory summary (what persists, appeared, or left) |
| **DET** | Detected objects with counts |

The top bar turns **green** when the narration is consistent, **red** when a contradiction is detected and a retry is happening.

---

## ⚙️ Configuration & Tuning

All key settings are at the top of `main.py`:

```python
LLM_MODEL = "llama-3.1-8b-instant"   # Groq model to use
model = YOLO("yolov8m.pt")            # YOLO model size
TemporalMemory(maxlen=9)              # How many frames to remember
```

### Change the YOLO model size

| Model | Speed | Accuracy |
|---|---|---|
| `yolov8n.pt` | Fastest | Lower |
| `yolov8s.pt` | Fast | Moderate |
| `yolov8m.pt` | Balanced (default) | Good |
| `yolov8l.pt` | Slower | High |
| `yolov8x.pt` | Slowest | Best |

Models are downloaded automatically on first run.

### Change pipeline frequency

In `run_system()`, find:

```python
if frame_id % 30 == 0:   # Run pipeline every 30 frames
```

Lower = more frequent updates (heavier on API calls), higher = less frequent.

### Change memory interval

```python
mem_agent = MemoryAgent(interval=1.5)   # Update memory summary every 1.5s
```

---

## 🔧 Troubleshooting

### Webcam not opening
- Make sure no other app is using the camera
- Try changing the device index: `cv2.VideoCapture(1)` instead of `0`

### `ModuleNotFoundError`
- Make sure your virtual environment is activated
- Re-run: `pip install opencv-python pillow numpy ultralytics transformers groq langgraph`

### Groq API errors (`401 Unauthorized`)
- Double-check your API key is correct and not expired
- Verify at [https://console.groq.com](https://console.groq.com)

### Slow performance / low FPS
- Switch to a smaller YOLO model: `yolov8n.pt`
- Reduce image size in VisionAgent: `imgsz=256` instead of `320`
- Use a machine with a GPU

### BLIP model download takes a long time
- This is normal on first run — BLIP downloads ~1 GB of weights
- They are cached locally after the first download

### Window doesn't open / display issues
- On some Linux systems, OpenCV GUI requires: `pip install opencv-python-headless` → `pip install opencv-python`
- Make sure a display is available (X server on Linux)

---

## 📁 Project Structure

```
perception-system/
│
├── main.py               ← All code lives here
├── yolov8m.pt            ← Auto-downloaded on first run
└── README.md
```

Models are cached by their respective libraries (Ultralytics cache for YOLO, Hugging Face cache for BLIP).

---

## 🧩 Architecture Overview

```
Frame Input
    │
    ▼
VisionAgent  →  YOLOv8 (object detection)
    │
    ▼
ContextAgent  →  BLIP (image captioning)
    │
    ▼
LanguageAgent  →  Groq LLM (narration)
    │
    ▼
CriticAgent  →  Groq LLM (consistency check)
    │
    ├── inconsistent + retry < 2  →  back to ContextAgent
    └── consistent or retry ≥ 2  →  END
```

All agents run as a compiled **LangGraph** state machine. A separate **MemoryAgent** thread runs every 1.5 seconds and tracks object trends across frames.

---

## 💡 Tips for Best Results

- Use good lighting for accurate YOLO detection
- A stable camera angle helps the memory agent detect meaningful changes
- If narrations feel repetitive, increase pipeline frequency (lower the `% 30` value)
- For video files, 30fps source videos give the best experience

---

## 📄 License

This project is provided for educational and personal use. Model weights (YOLOv8, BLIP) and API usage are subject to their respective licenses and terms of service.
