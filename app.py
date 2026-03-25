"""
FastAPI Prediction Server — Visionary AI
=========================================
Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import base64

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMG_SIZE         = 300
MODEL_PATH       = "best_model.keras"
CLASS_NAMES_PATH = "class_names.json"

model            = None
class_names      = []
LAST_CONV_NAME   = None
conv_owner       = None   # the sub-model that owns the last conv layer


def find_last_conv_layer(m):
    """
    Recursively walk the model tree.
    Returns (owner_submodel, layer) for the LAST Conv2D found.
    """
    result = None
    for layer in m.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            result = (m, layer)
        elif hasattr(layer, 'layers'):
            sub = find_last_conv_layer(layer)
            if sub is not None:
                result = sub
    return result


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, LAST_CONV_NAME, conv_owner

    print("Loading model...")
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"✅ Model loaded — {len(class_names)} classes")

        print("\n── Top-level model layers ──")
        for layer in model.layers:
            print(f"  [{layer.__class__.__name__:25s}]  {layer.name}")

        found = find_last_conv_layer(model)
        if found:
            conv_owner, conv_layer = found
            LAST_CONV_NAME = conv_layer.name
            print(f"\n✅ Grad-CAM layer : '{LAST_CONV_NAME}' in '{conv_owner.name}'")
        else:
            print("⚠️  No Conv2D found — Grad-CAM disabled")

    except FileNotFoundError as e:
        print(f"Model file not found: {e}")

    yield
    print("Shutting down")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Visionary AI — Eye Disease Detection",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def generate_gradcam_b64(image_bytes: bytes, pred_idx: int) -> str:
    if model is None or LAST_CONV_NAME is None or conv_owner is None:
        print("Grad-CAM skipped: not ready")
        return ""

    try:
        img_array  = preprocess(image_bytes)
        img_tensor = tf.constant(img_array, dtype=tf.float32)

        # Build a model: conv_owner.input → conv_layer.output
        # This stays entirely within the sub-model's own graph — no KeyError
        conv_layer        = conv_owner.get_layer(LAST_CONV_NAME)
        feature_map_model = tf.keras.Model(
            inputs  = conv_owner.input,
            outputs = conv_layer.output,
            name    = "gradcam_feature_extractor"
        )

        # Run feature extractor, then full model, watch the feature maps
        with tf.GradientTape() as tape:
            feature_maps = feature_map_model(img_tensor, training=False)
            tape.watch(feature_maps)
            predictions  = model(img_tensor, training=False)
            class_score  = predictions[:, pred_idx]

        grads = tape.gradient(class_score, feature_maps)

        if grads is None:
            print("Grad-CAM: gradients are None — non-differentiable path")
            return ""

        # Pool grads → weight channels → mean → heatmap
        pooled_grads    = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        feature_maps_np = feature_maps[0].numpy()                    # (H, W, C)
        weighted        = feature_maps_np * pooled_grads[np.newaxis, np.newaxis, :]
        heatmap         = np.mean(weighted, axis=-1)                 # (H, W)
        heatmap         = np.maximum(heatmap, 0)
        if heatmap.max() > 1e-8:
            heatmap /= heatmap.max()

        # Resize + colormap + blend
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        original        = np.array(
            Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        )
        original_bgr    = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        overlay         = cv2.addWeighted(original_bgr, 0.55, heatmap_colored, 0.45, 0)

        _, buf = cv2.imencode(".png", overlay)
        b64    = base64.b64encode(buf).decode("utf-8")
        print(f"✅ Grad-CAM OK — b64 length: {len(b64)}")
        return b64

    except Exception as exc:
        import traceback
        print(f"Grad-CAM failed: {exc}")
        traceback.print_exc()
        return ""


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":        "ok",
        "model_loaded":  model is not None,
        "gradcam_layer": LAST_CONV_NAME,
        "num_classes":   len(class_names),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    start    = time.time()
    arr      = preprocess(image_bytes)
    preds    = model.predict(arr, verbose=0)[0]
    elapsed  = round((time.time() - start) * 1000, 1)

    top3_idx = np.argsort(preds)[::-1][:3]
    top1_idx = int(top3_idx[0])

    return JSONResponse({
        "prediction":             class_names[top1_idx],
        "confidence":             round(float(preds[top1_idx]) * 100, 2),
        "inference_time_ms":      elapsed,
        "top3_predictions": [
            {
                "rank":               i + 1,
                "class":              class_names[int(top3_idx[i])],
                "confidence_percent": round(float(preds[top3_idx[i]]) * 100, 2),
            }
            for i in range(3)
        ],
        "gradcam_overlay_base64": generate_gradcam_b64(image_bytes, top1_idx),
    })


@app.get("/classes")
def get_classes():
    return {"total": len(class_names), "classes": class_names}