# 📘 PCB Defect Detection — Web App (Flask + OpenCV)

A modern and user-friendly web application for detecting defects in PCB images using image alignment, subtraction, and OpenCV-based processing.  
Includes a professional soft-colored UI, drag-and-drop uploads, live preview, AJAX processing, and downloadable results.

---

## 🚀 Features

- 📤 Upload PCB image **or** provide image URL  
- 🧠 PCB validation heuristic (green detection + edge density)  
- 🎯 ORB keypoint matching + homography alignment  
- 🔍 Template subtraction defect detection  
- 🟥 Red bounding boxes & overlays on defective regions  
- 📸 Live image preview  
- ⏳ AJAX progress bar  
- 💾 One-click result downloads  
- 🌈 Professional soft UI (dark, modern, responsive)

---

## 📂 Project Structure

```text
.
├── app.py                  # Flask backend (OpenCV processing)
├── README.md               # Documentation (this file)
├── templates/
│   └── index.html          # Enhanced frontend UI
├── templates_db/
│   └── template1.jpg       # Non-defective PCB template (add manually)
├── static/
│   ├── uploads/            # Uploaded images
│   └── results/            # Processed output (overlay + diff)
└── requirements.txt        # (Optional) dependency list
🔧 Installation & Setup
1️⃣ Create a Virtual Environment
bash
Copy code
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows
2️⃣ Install Dependencies
bash
Copy code
pip install flask numpy requests opencv-python-headless
Use opencv-python instead of headless if you need GUI windows.

3️⃣ Add Template Image
Place a clean, non-defective PCB image as:

bash
Copy code
templates_db/template1.jpg
4️⃣ Run the Application
bash
Copy code
python app.py
Now open:

cpp
Copy code
http://127.0.0.1:5000
🧠 How Defect Detection Works
User uploads a PCB image or enters a URL

App verifies if it looks like a PCB

Loads template image from templates_db/

Aligns uploaded image to template using ORB + homography

Converts both images to grayscale → blur → adaptive threshold

Performs absolute subtraction

Cleans results using morphological filters

Extracts contours representing defects

Draws red overlay + bounding boxes

Saves results to static/results/

Returns result to UI (HTML or AJAX)

🧪 API Endpoints
POST /
Traditional form submission — returns rendered HTML.

POST /analyze
AJAX endpoint — returns JSON.

Example JSON Response
json
Copy code
{
  "ok": true,
  "message": "Analysis complete",
  "result_image": "http://127.0.0.1:5000/static/results/test_result.png",
  "diff_image": "http://127.0.0.1:5000/static/results/test_diff.png",
  "defects": 3,
  "result_name": "test_result.png",
  "diff_name": "test_diff.png"
}
GET /download?name=<filename>
Downloads processed result.

🔧 Tuning Controls
In detect_defects_by_subtraction():
Parameter	Purpose
min_area	Ignore tiny noise blobs
max_area	Prevent very large false positives
Morphology kernel	Controls cleanup strength

PCB Detection Heuristic:
green_fraction > 0.02

edge_fraction > 0.02

Adjust these depending on dataset quality.

🐞 Troubleshooting
❌ Result Image Not Visible
If logs show:

sql
Copy code
GET /static/results\file.png 404
Windows-style backslashes were used.

Fix already implemented:
Static URLs use forward slashes:

python
Copy code
url_for('static', filename=f"results/{result_name}")
❌ “No template found”
Add:

bash
Copy code
templates_db/template1.jpg
❌ Alignment failure
Image too blurry

Wrong PCB model

Not enough keypoints

Increase ORB sensitivity:

python
Copy code
orb = cv2.ORB_create(nfeatures=8000)
🌈 UI Features
Drag & Drop file upload

Real-time preview

Upload progress bar

Soft dark theme

Responsive grid layout

Instant defect visualization

🔮 Future Enhancements
YOLO-based defect detection

Multi-template auto-selection

GPU acceleration (OpenCV CUDA)

Interactive parameter sliders

Full REST API with token auth

Docker deployment

