# 🎯 AI Object Detection Web App | YOLO26 Nano & Streamlit

An **AI-powered Object Detection Web Application** built using **Python, Streamlit, and Ultralytics YOLO26 Nano**. This application detects multiple objects in images in real time, identifies object classes, calculates confidence scores, and displays bounding boxes with detailed detection information.

Upload an image, adjust the confidence threshold, and let the YOLO26 Nano model identify objects through a simple and interactive web interface.

## 🚀 Key Features

- **Multiple Object Detection:** Detect multiple objects within a single image using YOLO26 Nano.
- **Image Upload:** Supports JPG, JPEG, and PNG image formats.
- **Adjustable Confidence Threshold:** Control detection accuracy using an interactive sidebar slider.
- **Bounding Box Visualization:** Display detected objects with bounding boxes, class labels, and confidence scores.
- **Detection Summary:** View the total number of detected objects and unique object categories.
- **Detailed Detection Results:** Explore object classes, confidence percentages, and bounding box coordinates.
- **Side-by-Side Image Comparison:** Compare original images with AI-annotated detection results.
- **Interactive Web Interface:** Easily perform object detection without writing code.

## 📸 Application Demo

The application provides three main sections:

| Image Upload & Settings | Detection Results | Detection Details |
|---|---|---|
| Upload images and configure confidence | View detected objects and bounding boxes | Explore detection metrics and coordinates |

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| YOLO26 Nano | Deep learning object detection |
| Ultralytics | YOLO model implementation |
| Pillow (PIL) | Image processing |
| Pandas | Detection results and data tables |
| NumPy | Numerical operations |

## 🧠 Object Detection Model

This project uses **YOLO26 Nano (yolo26n)** from Ultralytics.

YOLO26 Nano is a lightweight object detection model designed for efficient inference while maintaining strong detection performance.

The model identifies objects and provides:
- Object class labels
- Confidence scores
- Bounding box coordinates (X1, Y1, X2, Y2)

## 📁 Project Structure

```text
AI-Object-Detection/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
│
├── screenshots/
│   ├── upload.png
│   ├── results.png
│   └── details.png
│
└── README.md              # Project documentation
```

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Ashrif-aham/AI-Image-Classfier.git
```

```bash
cd AI-Image-Classfier
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
ultralytics
pillow
pandas
numpy
```

### 4. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal:

```text
http://localhost:8501
```

## 📖 How to Use

1. Launch the Streamlit application.
2. Upload an image in JPG, JPEG, or PNG format.
3. Adjust the minimum confidence threshold using the sidebar slider.
4. Click the **Detect Objects** button.
5. Wait for the YOLO26 Nano model to process the image.
6. View the original image and the annotated detection output.
7. Explore the detection summary and detailed results table.


## 🎚️ Application Configuration

| Parameter | Description | Default |
|---|---|---|
| Confidence Threshold | Minimum confidence required for a detection | `0.25` |
| Detection Model | YOLO26 Nano | `yolo26n` |
| Image Formats | Supported upload formats | JPG, JPEG, PNG |

Increasing the confidence threshold filters out lower-confidence detections. Lowering it allows more potential objects to be detected.

## 🔮 Future Enhancements

- [ ] Real-time object detection using a webcam.
- [ ] Live video object detection.
- [ ] Batch image processing.
- [ ] Download annotated images.
- [ ] Export detection results as CSV.
- [ ] Support multiple YOLO model versions.
- [ ] Object category filtering.
- [ ] Detection history and analytics dashboard.

## 🤝 Contributing

Contributions are welcome!

If you have suggestions, improvements, or feature requests, feel free to:
1. Fork this repository.
2. Create a new feature branch.
3. Commit your changes.
4. Submit a pull request.

## 📄 License

This project is licensed under the MIT License.

**Note:** Ultralytics models and software are subject to their respective licensing terms. Review the [Ultralytics Licensing](https://www.ultralytics.com/license) before using this project commercially.

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) – YOLO object detection models.
- [Streamlit](https://streamlit.io/) – Interactive web application framework.
- [Python](https://www.python.org/) – Programming language.
- [Pandas](https://pandas.pydata.org/) – Data manipulation and analysis.

## 👨‍💻 Author

**Ashrif Ahamed**

- GitHub: [@Ashrif-aham](https://github.com/Ashrif-aham)
- LinkedIn: [Ashrif Ahamed](https://www.linkedin.com/in/ashrif-ahamed/)

---

⭐ If you find this project useful, consider giving it a star!

**Keywords:** AI Object Detection, YOLO26 Nano, YOLO Object Detection, Computer Vision, Deep Learning, Python, Streamlit, Image Processing, Ultralytics, Artificial Intelligence, Machine Learning, Real-Time Object Detection.
