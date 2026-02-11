# Nepali Plate Recognition using Machine Learning

A ML–based Automatic Number Plate Recognition (ANPR) system designed to accurately recognize multilingual vehicle registration plates, including English alphabets, English digits, Nepali digits, and compound Nepali characters. Unlike conventional ANPR systems that are primarily limited to Latin scripts, this system addresses the additional complexity introduced by compound glyph structures and multilingual formatting found in Nepali number plates.

The proposed framework implements a structured, modular detection-to-recognition pipeline. Character regions are first localized using YOLO-format bounding box annotations, enabling precise extraction of individual characters from full plate images. These cropped character images are then processed through a Convolutional Neural Network (CNN) trained for multi-class character classification. The architecture ensures robust feature extraction, spatial invariance, and accurate classification across diverse character sets.

By integrating object localization and deep learning–based classification, this project provides a scalable and extensible foundation for multilingual ANPR systems suitable for intelligent transportation and smart surveillance applications.

---

# Introduction

Automatic Number Plate Recognition (ANPR) is a computer vision application that detects and extracts vehicle registration numbers from images or video streams.

Unlike traditional ANPR systems that focus only on English characters, this project supports:

- English Alphabets (A–Z)
- English Digits (0–9)
- Nepali Digits
- Compound Nepali Characters (e.g., का, ना, को, बा)

The project is designed as a research-oriented, modular ANPR system that can be extended into real-time applications.

---

# Problem Statement

Nepali number plates present unique challenges:

- Multilingual structure
- Compound characters
- Diverse font styles
- Limited publicly available datasets

Most existing ANPR implementations do not support compound Nepali characters.  
This project addresses that gap using a CNN-based character classification approach.

---

# System Architecture

The system follows a two-stage pipeline:

Full Plate Image
 
↓

Character Detection (YOLO Label Boxes)

↓

Character Cropping & Preprocessing

↓

CNN Character Classification

↓

Left-to-Right Character Concatenation

↓

Final Recognized Plate Number

---

# Dataset Structure

## Character-Level Dataset

<img width="270" height="404" alt="Screenshot 2026-02-11 102310" src="https://github.com/user-attachments/assets/6014ea01-e05c-4305-9876-dc6549ec4a0a" />
<img width="283" height="87" alt="Screenshot 2026-02-11 103025" src="https://github.com/user-attachments/assets/67c189a1-5953-447e-995d-ba247466fb4e" />


- Each folder represents a single character class
- Grayscale images
- Resized to 48×48
- Total Classes: 53
- Loaded Images: 30,114

---

## Full Plate Dataset

<img width="253" height="254" alt="Screenshot 2026-02-11 102352" src="https://github.com/user-attachments/assets/5d8e4800-f0c4-4f6d-b32c-d70c48f3ba95" />

---

# Data Preprocessing

### Character Dataset Processing
- Loaded using PIL (Unicode safe)
- Converted to grayscale
- Corrupted files skipped
- Resized to 48×48
- Normalized to range [0,1]
- Labels converted to one-hot encoding

### Plate Image Processing
- YOLO bounding boxes converted to pixel coordinates
- Character regions cropped
- Resized and normalized
- Sorted by x-coordinate (left to right)

---

# Model Architecture

<img width="787" height="791" alt="Model" src="https://github.com/user-attachments/assets/992f0110-9a5b-4cdb-92c2-00be86477fc1" />

---

# Training Configuration

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metric: Accuracy
- Epochs: 15
- Batch Size: 64
- Train/Test Split: 80% / 20%

---

# Results

### Training Visualizations

Model Performance:

<img width="1189" height="390" alt="Output" src="https://github.com/user-attachments/assets/5741f76b-2acb-451f-a090-4c01daaf8a87" />

Character Recognition:

<img width="1063" height="304" alt="CharRecognition" src="https://github.com/user-attachments/assets/57ec0b7d-9f2f-4801-8dae-90642dc510d3" />

<img width="1037" height="264" alt="Screenshot 2026-02-11 102744" src="https://github.com/user-attachments/assets/c0dde2f5-3d49-4b2a-9f7a-920ca58e5782" />


Character Detection:

<img width="640" height="230" alt="CharDetection" src="https://github.com/user-attachments/assets/d45fb9df-2f23-46b4-9ed4-f3fedb5a723c" />

<img width="728" height="498" alt="Screenshot 2026-02-11 102726" src="https://github.com/user-attachments/assets/9aeb4a4d-79cb-4fc1-9a14-3c17837c940b" />


Confusion Matrix:

<img width="576" height="519" alt="ConfusionMatrix" src="https://github.com/user-attachments/assets/388d4cb9-a1e4-46d5-bdb2-ffb50c2ebd90" />


---

# Applications

This system can be used in:

- Traffic Monitoring Systems
- Smart City Infrastructure
- Toll Booth Automation
- Parking Management Systems
- Law Enforcement Vehicle Tracking
- CCTV Surveillance Systems
- Automated Entry & Exit Systems

---

# Limitations

- Detection depends on YOLO label files
- Not implemented for real-time video yet
- Sensitive to motion blur and extreme lighting
- Compound characters require larger dataset for higher accuracy
- Limited dataset size may affect generalization

---

# Future Improvements

- Integrate YOLOv8 for automatic character detection
- Implement CRNN + CTC for sequence recognition
- Add real-time video processing
- Apply data augmentation
- Deploy as web application
- Improve compound Nepali character accuracy

---

# Conclusion

This project presents a robust and scalable multilingual Automatic Number Plate Recognition (ANPR) system built using deep learning methodologies. By integrating structured character localization with CNN-based classification, the system successfully addresses the complexities associated with recognizing English characters, Nepali digits, and compound Nepali characters within a unified framework.

Unlike traditional ANPR implementations that are limited to single-language datasets, this solution demonstrates the feasibility of deploying multilingual recognition models in real-world transportation environments. The modular architecture ensures flexibility — allowing seamless integration with detection models, surveillance systems, or API-based deployment pipelines.

From a technical perspective, the project establishes a reliable detection-to-recognition workflow that can serve as a foundation for further research and commercial applications. From an industry standpoint, the system offers practical value for smart city infrastructure, automated toll systems, parking automation, vehicle access control, and law enforcement monitoring.

While future improvements such as real-time inference optimization, automatic detection integration, and dataset expansion can further enhance performance, the current implementation provides a solid, scalable baseline for multilingual ANPR systems tailored to regional requirements.

Overall, this work demonstrates how deep learning can be effectively applied to localized transportation challenges, bridging the gap between academic research and industry-ready intelligent vehicle recognition solutions.
