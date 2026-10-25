<h1 align="center">🚗 Nepali Number Plate Recognition</h1>

<p align="center">Reads Nepali registration plates character by character: English letters, English and Nepali digits, and compound Devanagari characters such as का, ना, को and बा.</p>

<p align="center">
  <img src="https://img.shields.io/badge/model-Keras%20CNN-ff6f00" alt="CNN">
  <img src="https://img.shields.io/badge/classes-53-blue" alt="53 classes">
  <img src="https://img.shields.io/badge/images-30%2C114-brightgreen" alt="30k images">
  <img src="https://img.shields.io/badge/script-Devanagari%20%2B%20Latin-9b59b6" alt="Scripts">
</p>

<table>
  <tr>
    <td width="45%"><img src="https://github.com/user-attachments/assets/d45fb9df-2f23-46b4-9ed4-f3fedb5a723c" alt="Characters located"><br><sub><b>Characters located</b> (YOLO boxes)</sub></td>
    <td width="55%"><img src="https://github.com/user-attachments/assets/57ec0b7d-9f2f-4801-8dae-90642dc510d3" alt="Recognised"><br><sub><b>Each crop classified</b></sub></td>
  </tr>
</table>

## How it works

```mermaid
flowchart LR
    P["Plate image"] --> B["YOLO label boxes<br/>→ pixel coords"]
    B --> C["Crop each character<br/>48×48 grey · 0–1"]
    C --> N["CNN · 53 classes"]
    N --> S["Sort by x<br/>left → right"]
    S --> O["Plate string"]
```

| | |
|---|---|
| **Character set** | 53 classes, one folder each, 30 114 images, loaded with PIL so Unicode folder names work |
| **Training** | Adam · categorical cross-entropy · 15 epochs · batch 64 · 80 / 20 split → `char_cnn.h5` |
| **Plates** | Full plate images with YOLO-format `.txt` labels |

<table>
  <tr>
    <td width="60%"><img src="https://github.com/user-attachments/assets/5741f76b-2acb-451f-a090-4c01daaf8a87" alt="Training"><br><sub><b>Training curves</b></sub></td>
    <td width="40%"><img src="https://github.com/user-attachments/assets/388d4cb9-a1e4-46d5-bdb2-ffb50c2ebd90" alt="Confusion"><br><sub><b>Confusion matrix</b></sub></td>
  </tr>
</table>

<details>
<summary><b>Dataset samples, model summary and more results</b></summary>
<br>
<img src="https://github.com/user-attachments/assets/6014ea01-e05c-4305-9876-dc6549ec4a0a" width="220" alt="Char folders">
<img src="https://github.com/user-attachments/assets/5d8e4800-f0c4-4f6d-b32c-d70c48f3ba95" width="220" alt="Plate dataset">
<img src="https://github.com/user-attachments/assets/67c189a1-5953-447e-995d-ba247466fb4e" width="280" alt="Char samples"><br>
<img src="https://github.com/user-attachments/assets/992f0110-9a5b-4cdb-92c2-00be86477fc1" width="480" alt="Model"><br>
<img src="https://github.com/user-attachments/assets/c0dde2f5-3d49-4b2a-9f7a-920ca58e5782" width="640" alt="Recognition 2"><br>
<img src="https://github.com/user-attachments/assets/9aeb4a4d-79cb-4fc1-9a14-3c17837c940b" width="480" alt="Detection 2">
</details>

## Run it

```bash
pip install tensorflow opencv-python pillow numpy matplotlib scikit-learn
jupyter notebook Main2.ipynb
```

Set `CHAR_DATASET`, `MODEL_DIR`, `PLATE_IMAGES` and `LABELS_PATH` (currently `E:\NumberPlateRecognition\...`) at the top.

## Status & known issues

| Issue | Effect |
|---|---|
| **Character boxes come from ground-truth YOLO label files, not a detector** | This is recognition only. A new photo without a `.txt` label can't be read |
| Characters are sorted by x only | Two-row plates (common in Nepal) come out with the rows interleaved |
| The notebook has no saved outputs, and no test accuracy is stated | The only results are the images above |
| Dataset and `char_cnn.h5` not in the repo | Training and inference can't be reproduced as cloned |
| Per-character classification | No language model, so errors such as a `0` vs `०` mix-up aren't caught |

## Roadmap

- [ ] Train a YOLOv8 character / plate detector to make it end to end
- [ ] Row clustering (by y) before the left-to-right sort
- [ ] Report test accuracy and per-class F1; augment rare compound characters
- [ ] CRNN + CTC for whole-plate reading; real-time video demo
