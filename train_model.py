"""
Visionary AI - Eye Disease Detection from Fundus Images
=======================================================
Dataset: 997 images, 39 classes (80/20 split → 798 train / 199 val)
Model: EfficientNetB3 + custom head (best accuracy/speed tradeoff)
Extras: Grad-CAM, augmentation, LR scheduling, early stopping
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import json

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE    = 300          # EfficientNetB3 native size
BATCH_SIZE  = 16           # small dataset → small batch
EPOCHS      = 60
NUM_CLASSES = 39
SEED        = 42
DATA_DIR    = r"C:\Users\Lenovo\Downloads\files cloude\dataset"  # <-- change to your dataset root path
MODEL_PATH  = "best_model.keras"
CLASS_NAMES_PATH = "class_names.json"

# ─────────────────────────────────────────────
# 2. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
def build_datasets(data_dir):
    """Load train/val splits with aggressive augmentation on train."""

    # Training dataset with augmentation
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    print(f"\n✅ Classes ({len(class_names)}): {class_names}\n")

    # Save class names for FastAPI inference
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)

    # Augmentation layer (applied only to training)
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess_train(images, labels):
        images = augmentation(images, training=True)
        images = tf.cast(images, tf.float32)
        images = keras.applications.efficientnet.preprocess_input(images)
        return images, labels

    def preprocess_val(images, labels):
        images = tf.cast(images, tf.float32)
        images = keras.applications.efficientnet.preprocess_input(images)
        return images, labels

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(preprocess_val,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────
# 3. MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> Model:
    """
    EfficientNetB3 backbone (pretrained on ImageNet) +
    custom classification head with dropout for regularization.
    """
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Phase-1: freeze backbone
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="VisinaryAI_EfficientNetB3")
    model.summary()
    return model, base


# ─────────────────────────────────────────────
# 4. TRAINING — TWO-PHASE (TRANSFER + FINE-TUNE)
# ─────────────────────────────────────────────
def train(data_dir=DATA_DIR):
    train_ds, val_ds, class_names = build_datasets(data_dir)
    model, base = build_model(NUM_CLASSES)

    # ── Phase 1: Train head only ──
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    callbacks_phase1 = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        TensorBoard(log_dir="./logs/phase1"),
    ]

    print("\n🚀 Phase 1: Training classification head ...\n")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks_phase1,
    )

    # ── Phase 2: Fine-tune top layers of backbone ──
    base.trainable = True
    # Freeze first 60% of layers, fine-tune rest
    fine_tune_at = int(len(base.layers) * 0.6)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),   # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    callbacks_phase2 = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.3, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        TensorBoard(log_dir="./logs/phase2"),
    ]

    print("\n🔥 Phase 2: Fine-tuning backbone ...\n")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_phase2,
    )

    plot_history(history1, history2)
    evaluate(model, val_ds, class_names)
    return model, class_names


# ─────────────────────────────────────────────
# 5. EVALUATION & PLOTS
# ─────────────────────────────────────────────
def plot_history(h1, h2):
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    vloss= h1.history["val_loss"] + h2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(acc, label="Train Acc"); axes[0].plot(vacc, label="Val Acc")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(loss, label="Train Loss"); axes[1].plot(vloss, label="Val Loss")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("📊 Saved: training_curves.png")


def evaluate(model, val_ds, class_names):
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                annot=False, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("📊 Saved: confusion_matrix.png")


# ─────────────────────────────────────────────
# 6. GRAD-CAM (Explainability)
# ─────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    """Generate Grad-CAM heatmap for the predicted class."""
    grad_model = Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def apply_gradcam(img_path, model, class_names, save_path="gradcam_output.png"):
    """Load image, run Grad-CAM, save overlay."""
    img = keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.utils.img_to_array(img)
    img_preprocessed = keras.applications.efficientnet.preprocess_input(
        np.expand_dims(img_array.copy(), axis=0)
    )

    heatmap = make_gradcam_heatmap(img_preprocessed, model)
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    original = np.uint8(img_array)
    superimposed = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

    preds = model.predict(img_preprocessed, verbose=0)[0]
    top3_idx = np.argsort(preds)[::-1][:3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(heatmap_resized, cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(superimposed); axes[2].axis("off")
    axes[2].set_title(
        f"Top-1: {class_names[top3_idx[0]]} ({preds[top3_idx[0]]:.1%})\n"
        f"Top-2: {class_names[top3_idx[1]]} ({preds[top3_idx[1]]:.1%})\n"
        f"Top-3: {class_names[top3_idx[2]]} ({preds[top3_idx[2]]:.1%})"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"🔍 Grad-CAM saved: {save_path}")


# ─────────────────────────────────────────────
# 7. SINGLE IMAGE PREDICTION (for testing)
# ─────────────────────────────────────────────
def predict_image(img_path, model=None, class_names=None):
    if model is None:
        model = keras.models.load_model(MODEL_PATH)
    if class_names is None:
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)

    img = keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.utils.img_to_array(img)
    img_array = keras.applications.efficientnet.preprocess_input(
        np.expand_dims(img_array, axis=0)
    )
    preds = model.predict(img_array, verbose=0)[0]
    top3 = np.argsort(preds)[::-1][:3]

    result = {
        "prediction": class_names[top3[0]],
        "confidence": float(preds[top3[0]]),
        "top3": [
            {"class": class_names[i], "confidence": float(preds[i])} for i in top3
        ]
    }
    print(json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="Path to dataset root")
    parser.add_argument("--predict", default=None, help="Path to single image for prediction")
    parser.add_argument("--gradcam", default=None, help="Path to image for Grad-CAM")
    args = parser.parse_args()

    if args.predict:
        predict_image(args.predict)
    elif args.gradcam:
        model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        apply_gradcam(args.gradcam, model, class_names)
    else:
        train(args.data_dir)
