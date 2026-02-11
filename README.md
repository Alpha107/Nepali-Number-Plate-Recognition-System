Nepali & English Number Plate Recognition using Deep Learning

A deep learning–based Automatic Number Plate Recognition (ANPR) system capable of recognizing English, Nepali digits, and compound Nepali characters from vehicle number plates.

This project implements a complete detection-to-recognition pipeline using YOLO-style bounding boxes and a Convolutional Neural Network (CNN) for character classification.

📌 Table of Contents

Introduction

Why This Project

System Architecture

Dataset Structure

Preprocessing

Model Architecture

Training Configuration

Results & Visualizations

Project Structure

Installation

Usage

Applications

Limitations

Future Improvements

Conclusion

Author

🧠 Introduction

Automatic Number Plate Recognition (ANPR) is a computer vision application that detects and extracts vehicle registration numbers from images or video.

Unlike most ANPR systems that focus only on English characters, this project supports:

English alphabets (A–Z)

English digits (0–9)

Nepali digits

Compound Nepali characters (e.g., का, ना, को, बा, etc.)

The system is designed for academic research and can be extended for real-world traffic and smart city applications.

🎯 Why This Project

Nepali number plates contain:

Multilingual characters

Compound glyph structures

Unique formatting patterns

Most publicly available ANPR systems do not support Nepali compound characters.
This project bridges that gap using a CNN-based character recognition model.

🏗 System Architecture

The pipeline follows two main stages:

Full Plate Image
        ↓
Character Detection (YOLO Labels)
        ↓
Character Cropping & Normalization
        ↓
CNN Character Classification
        ↓
Final Plate Text Output

Stage 1: Character Detection

Uses YOLO-format label files

Extracts bounding boxes for each character

Stage 2: Character Recognition

Cropped characters resized to 48×48

Passed through trained CNN model

Predicted characters combined left-to-right

📂 Dataset Structure
1️⃣ Character-Level Training Dataset
MainDataset/
├── A/
├── B/
├── 0/
├── 1/
├── ka/
├── na/
├── ko/
├── बा/
├── न/
└── ...


Each folder represents one character class

Grayscale images

Resized to 48 × 48

Total Classes: 53

Loaded Images: 3567

2️⃣ Full Plate Dataset
Dataset/
├── Images/
│   ├── image1.jpg
│   └── image2.jpg
├── labels/
│   ├── image1.txt
│   └── image2.txt


YOLO Label Format:

class_id x_center y_center width height


These labels allow extraction of individual characters from number plate images.

⚙ Preprocessing
Character Dataset

Loaded using PIL (Unicode-safe)

Converted to grayscale

Corrupted files skipped

Resized to 48×48

Normalized to range [0,1]

Labels one-hot encoded

Plate Images

Read using OpenCV

Bounding boxes converted from YOLO format

Cropped characters resized and normalized

Sorted left-to-right before prediction

🤖 Model Architecture

Input: 48 × 48 × 1

CNN Layers:

Conv2D (32 filters, 3×3, ReLU)

MaxPooling (2×2)

Conv2D (64 filters, 3×3, ReLU)

MaxPooling (2×2)

Conv2D (128 filters, 3×3, ReLU)

Flatten

Dense (256 units, ReLU)

Dropout (0.5)

Dense (Softmax – 53 classes)

🏋️ Training Configuration

Optimizer: Adam

Loss: Categorical Crossentropy

Metric: Accuracy

Epochs: 15

Batch size: 64

Train/Test Split: 80/20

📊 Results & Visualizations
📌 Character Distribution

Add screenshot here

![Class Distribution](images/class_distribution.png)

📌 Training Samples

Add screenshot here

![Training Samples](images/training_samples.png)

📌 Accuracy vs Epoch

Add screenshot here

![Accuracy Plot](images/accuracy_plot.png)

📌 Loss vs Epoch

Add screenshot here

![Loss Plot](images/loss_plot.png)

📌 Confusion Matrix

Add screenshot here

![Confusion Matrix](images/confusion_matrix.png)

📌 Detection Example

Add screenshot here

![Detection](images/detection_boxes.png)

📌 Extracted Characters

Add screenshot here

![Extracted Characters](images/extracted_characters.png)

📌 Final Output Example

Example:

Recognized Number Plate: बा ३४५६


Add screenshot here

![Final Output](images/final_output.png)

📁 Project Structure
NumberPlateRecognition/
│
├── MainDataset/
├── Dataset/
│   ├── Images/
│   ├── labels/
│
├── models/
│   └── char_cnn.h5
│
├── images/   # Visualization screenshots
│
├── train.py
├── detect_and_recognize.py
└── README.md

🛠 Installation

Install required libraries:

pip install tensorflow opencv-python pillow matplotlib scikit-learn

▶ Usage
1️⃣ Train Model

Run training script to generate:

models/char_cnn.h5

2️⃣ Test on Image

Provide:

Plate image path

Corresponding YOLO label file

recognized = recognize_plate_from_path(image_path, label_path)
print("Recognized Plate:", recognized)

🌍 Applications

Traffic Monitoring Systems

Smart Cities

Toll Booth Automation

Parking Management

Law Enforcement

Vehicle Access Control Systems

CCTV Surveillance

⚠ Limitations

Depends on YOLO label files for detection

Not real-time yet

Performance depends on dataset size

Sensitive to blur and extreme lighting

Compound Nepali characters require larger datasets for high accuracy

🚀 Future Improvements

Integrate YOLOv8 for automatic detection

Replace CNN with CRNN + CTC for sequence recognition

Real-time video processing

Data augmentation for robustness

Web app deployment

Mobile-optimized inference model

Improve compound character accuracy

🧾 Conclusion

This project demonstrates a complete multilingual ANPR system using deep learning. It successfully handles English and Nepali compound characters using a CNN-based classification pipeline.

The modular design allows easy extension to real-time systems and further research applications.

This project forms a strong foundation for intelligent transportation and smart surveillance systems.
