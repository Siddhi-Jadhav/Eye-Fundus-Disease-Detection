# 👁️ Visionary AI — Eye Disease Detection from Fundus Images
**C-DAC Hackathon | 997 images | 39 classes**

---

## 📁 Project Structure
```
visionary-ai/
├── train_model.py       ← Training script (EfficientNetB3 + Grad-CAM)
├── app.py               ← FastAPI prediction server
├── requirements.txt     ← All dependencies
├── best_model.keras     ← Saved after training (auto-generated)
├── class_names.json     ← Class list (auto-generated)
├── training_curves.png  ← Accuracy/loss plots (auto-generated)
└── confusion_matrix.png ← Confusion matrix (auto-generated)
```

---

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your dataset path in train_model.py
DATA_DIR = "./dataset"   # folder with 39 sub-folders (one per class)
```

### Dataset folder structure expected:
```
dataset/
├── Diabetic_Retinopathy_Mild/
│   ├── img001.jpg
│   └── ...
├── Glaucoma/
│   ├── img001.jpg
│   └── ...
└── ... (39 folders total)
```

---

## 🚀 Training

```bash
python train_model.py --data_dir ./dataset
```

This runs **two phases**:
1. **Phase 1** — Train only the custom head (20 epochs, LR=1e-3)
2. **Phase 2** — Fine-tune top 40% of EfficientNetB3 backbone (up to 60 epochs, LR=1e-4)

Early stopping prevents overfitting. Best weights saved to `best_model.keras`.

---

## 🔍 Test Single Image

```bash
python train_model.py --predict path/to/fundus.jpg
```

## 🔥 Grad-CAM Visualization

```bash
python train_model.py --gradcam path/to/fundus.jpg
```

---

## 🌐 Run FastAPI Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- **Swagger UI**: http://localhost:8000/docs
- **POST /predict** — Upload fundus image → get prediction + Grad-CAM
- **GET /classes** — List all 39 detectable classes

### Example API Response:
```json
{
  "prediction": "Diabetic Retinopathy - Moderate",
  "confidence": 87.34,
  "inference_time_ms": 142.5,
  "top3_predictions": [
    {"rank": 1, "class": "Diabetic Retinopathy - Moderate", "confidence_percent": 87.34},
    {"rank": 2, "class": "Diabetic Retinopathy - Mild",     "confidence_percent": 8.12},
    {"rank": 3, "class": "Normal",                          "confidence_percent": 2.54}
  ],
  "gradcam_overlay_base64": "..."
}
```

---

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Backbone | EfficientNetB3 (ImageNet pretrained) |
| Input size | 300×300 RGB |
| Head | GAP → BN → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax(39) |
| Augmentation | Flip, Rotate, Zoom, Contrast, Brightness, Translate |
| Explainability | Grad-CAM on `top_conv` layer |

---

## 📊 Judging Criteria Coverage

| Criteria | How covered |
|---|---|
| Model Accuracy (15) | EfficientNetB3 + 2-phase fine-tuning |
| Robustness (10) | Heavy augmentation + Dropout |
| Explainability (10) | Grad-CAM overlay included in API response |
| Preprocessing (10) | EfficientNet normalize + augmentation pipeline |
| Inference Speed (5) | ~100-200ms on CPU; faster on GPU |
| Frontend/UX (10) | FastAPI Swagger UI + JSON response |
| Innovative Features (10) | Top-3 predictions, Grad-CAM in API |
| Deployment (10) | FastAPI + Uvicorn, Docker-ready |
