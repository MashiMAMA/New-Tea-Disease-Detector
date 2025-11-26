# pyright: reportMissingImports=false
# 2 processing report

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import io
import json
import platform
import datetime

# === Directories ===
BASE_DIR = r"D:\Research\5 Preprocessed\Leaf_Disease_Dataset6"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VAL_DIR = os.path.join(BASE_DIR, "Validation")
TEST_DIR = os.path.join(BASE_DIR, "Test")

# Where to save the report and small artifacts
REPORT_SAVE_DIR = r"D:\Research\5 Preprocessed"
REPORT_FILENAME = "Model_Training_Report_2.json"
REPORT_PATH = os.path.join(REPORT_SAVE_DIR, REPORT_FILENAME)
TRAIN_PLOT_PATH = os.path.join(REPORT_SAVE_DIR, "training_history_plot.png")
MODEL_SAVE_PATH = os.path.join(REPORT_SAVE_DIR, "tea_leaf_model5.h5")

os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

# === Constants ===
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 32

# === Helper utilities for report ===
def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

def folder_size_bytes(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                fp = os.path.join(root, f)
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total

def count_files_by_class(split_dir):
    counts = {}
    total = 0
    if not os.path.exists(split_dir):
        return counts, total
    for class_name in sorted(os.listdir(split_dir)):
        class_path = os.path.join(split_dir, class_name)
        if os.path.isdir(class_path):
            n = 0
            for root, _, files in os.walk(class_path):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        n += 1
            counts[class_name] = n
            total += n
    return counts, total

def model_summary_to_string(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    s = stream.getvalue()
    stream.close()
    return s

def get_env_info():
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__
    }
    try:
        gpus = tf.config.list_physical_devices('GPU')
        info['gpus'] = [str(g) for g in gpus]
    except Exception as e:
        info['gpus'] = f"error listing GPUs: {e}"
    return info

# === Load Datasets ===
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Detected Classes:", class_names)

# === Normalize RGB Images ===
def normalize_rgb(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(normalize_rgb).prefetch(AUTOTUNE)
val_ds = val_ds.map(normalize_rgb).prefetch(AUTOTUNE)
test_ds = test_ds.map(normalize_rgb).prefetch(AUTOTUNE)

# === Display Sample Images ===
plt.figure(figsize=(10, 8))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()

#  CNN MODEL
model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks 
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# Train Model
train_start_time = datetime.datetime.now().isoformat()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)
train_end_time = datetime.datetime.now().isoformat()

# === Evaluate ===
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# === Save Model ===
try:
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved as {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Warning: failed to save model: {e}")

# === Save training plot ===
try:
    hist = history.history
    plt.figure()
    if 'loss' in hist:
        plt.plot(hist['loss'], label='train_loss')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='val_loss')
    if 'accuracy' in hist:
        plt.plot(hist['accuracy'], label='train_acc')
    if 'val_accuracy' in hist:
        plt.plot(hist['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training history')
    plt.tight_layout()
    plt.savefig(TRAIN_PLOT_PATH)
    plt.close()
    print(f"Training plot saved to {TRAIN_PLOT_PATH}")
except Exception as e:
    print("Warning:", e)

# === Generate processing report JSON ===
report = {}

report['training_start_time'] = train_start_time
report['training_end_time'] = train_end_time
report['report_generated_time'] = datetime.datetime.now().isoformat()

report['environment'] = get_env_info()

# dataset stats
splits = {}
for split_name, split_dir in [("Train", TRAIN_DIR), ("Validation", VAL_DIR), ("Test", TEST_DIR)]:
    counts_by_class, total = count_files_by_class(split_dir)
    size_bytes = folder_size_bytes(split_dir)
    splits[split_name] = {
        "path": split_dir,
        "total_images": total,
        "folder_size_bytes": size_bytes,
        "folder_size_readable": sizeof_fmt(size_bytes),
        "per_class_counts": counts_by_class
    }

report['dataset'] = splits
report['class_names'] = class_names
report['model_summary_text'] = model_summary_to_string(model)
report['training_history'] = history.history
report['test'] = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc)
}
report['artifacts'] = {
    "model_path": MODEL_SAVE_PATH,
    "training_plot_path": TRAIN_PLOT_PATH
}

try:
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nProcessing report saved to: {REPORT_PATH}")
except Exception as e:
    print(f"Error saving report: {e}")


















# 2
# import tensorflow as tf
# from tensorflow import keras
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import os

# # === Directories ===
# BASE_DIR = r"D:\Research\4 Preprocessed\Leaf_Disease_Dataset5"
# TRAIN_DIR = os.path.join(BASE_DIR, "Train")
# VAL_DIR = os.path.join(BASE_DIR, "Validation")
# TEST_DIR = os.path.join(BASE_DIR, "Test")

# # === Constants ===
# IMG_SIZE = (256, 256)
# BATCH_SIZE = 32
# SEED = 42

# # === Load Datasets ===
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TRAIN_DIR,
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     VAL_DIR,
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     seed=SEED,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# class_names = train_ds.class_names
# print("Detected Classes:", class_names)

# # === Normalize RGB Images ===
# def normalize_rgb(image, label):
#     image = tf.cast(image, tf.float32) / 255.0
#     return image, label

# train_ds = train_ds.map(normalize_rgb)
# val_ds = val_ds.map(normalize_rgb)
# test_ds = test_ds.map(normalize_rgb)

# # === Display Sample Images from Train Set ===
# plt.figure(figsize=(10, 8))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy())
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.tight_layout()
# plt.show()

# # === Data Augmentation ===
# data_augmentation = keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
# ])

# # === Define CNN Model ===
# model = models.Sequential([
#     layers.Input(shape=(256, 256, 3)),
#     data_augmentation,

#     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(2, 2),

#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(2, 2),

#     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(2, 2),

#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.4),
#     layers.Dense(len(class_names), activation='softmax')
# ])

# # === Compile Model ===
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # === Callbacks ===
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
# ]

# # === Train Model ===
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=32,
#     callbacks=callbacks
# )

# # === Evaluate on Test Set ===
# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test Accuracy: {test_acc:.4f}")

# # === Save Model ===
# model.save("tea_leaf_model5.h5")
# print("Model saved as tea_leaf_model5.h5")













# # 1
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# import os

# # =========================
# # Paths
# # =========================
# BASE_DIR = r"D:\Research\5 Preprocessed\Leaf_Disease_Dataset6"
# TRAIN_DIR = os.path.join(BASE_DIR, "Train")
# VAL_DIR = os.path.join(BASE_DIR, "Validation")
# TEST_DIR = os.path.join(BASE_DIR, "Test")

# # =========================
# # Constants
# # =========================
# IMG_SIZE = (256, 256)
# BATCH_SIZE = 32

# # =========================
# # Load datasets (No splitting)
# # =========================
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TRAIN_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     color_mode="rgb"  # still RGB here, we'll convert later
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     VAL_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     color_mode="rgb"
# )

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     color_mode="rgb"
# )

# class_names = train_ds.class_names
# print("Detected Classes:", class_names)

# # =========================
# # Preprocessing: RGB → Grayscale → Normalize
# # =========================
# def to_grayscale_and_normalize(image, label):
#     image = tf.image.rgb_to_grayscale(image)   # (H, W, 1)
#     image = tf.cast(image, tf.float32) / 255.0 # Normalize to [0,1]
#     return image, label

# train_ds = train_ds.map(to_grayscale_and_normalize)
# val_ds = val_ds.map(to_grayscale_and_normalize)
# test_ds = test_ds.map(to_grayscale_and_normalize)

# # =========================
# # Plot sample grayscale images
# # =========================
# plt.figure(figsize=(10, 8))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().squeeze(), cmap='gray')
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.tight_layout()
# plt.show()

# # =========================
# # Model
# # =========================
# model = models.Sequential([
#     layers.Input(shape=(256, 256, 1)),

#     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(2, 2),

#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(2, 2),

#     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(2, 2),

#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(len(class_names), activation='softmax')
# ])

# # =========================
# # Compile & Train
# # =========================
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=32
# )

# # =========================
# # Evaluate on Test set
# # =========================
# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test Accuracy: {test_acc:.4f}")



























# # 1 Processing report
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# import os
# import time
# import datetime
# import platform
# import io
# import json

# # =========================
# # Paths
# # =========================
# BASE_DIR = r"D:\Research\5 Preprocessed\Leaf_Disease_Dataset6"
# TRAIN_DIR = os.path.join(BASE_DIR, "Train")
# VAL_DIR = os.path.join(BASE_DIR, "Validation")
# TEST_DIR = os.path.join(BASE_DIR, "Test")

# # Report path (requested)
# REPORT_SAVE_DIR = r"D:\Research\5 Preprocessed"
# REPORT_FILENAME = "Model_Training_Report_1.json"
# REPORT_PATH = os.path.join(REPORT_SAVE_DIR, REPORT_FILENAME)
# os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

# # =========================
# # Constants
# # =========================
# IMG_SIZE = (256, 256)
# BATCH_SIZE = 32
# EPOCHS = 32

# # =========================
# # Helper utilities
# # =========================
# def sizeof_fmt(num, suffix='B'):
#     for unit in ['','K','M','G','T']:
#         if abs(num) < 1024.0:
#             return f"{num:3.1f}{unit}{suffix}"
#         num /= 1024.0
#     return f"{num:.1f}P{suffix}"

# def folder_size_bytes(path):
#     total = 0
#     if not os.path.exists(path):
#         return 0
#     for root, dirs, files in os.walk(path):
#         for f in files:
#             try:
#                 fp = os.path.join(root, f)
#                 total += os.path.getsize(fp)
#             except Exception:
#                 pass
#     return total

# def count_files_by_class(split_dir):
#     counts = {}
#     total = 0
#     if not os.path.exists(split_dir):
#         return counts, total
#     for class_name in sorted(os.listdir(split_dir)):
#         class_path = os.path.join(split_dir, class_name)
#         if os.path.isdir(class_path):
#             n = 0
#             for root, _, files in os.walk(class_path):
#                 for f in files:
#                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
#                         n += 1
#             counts[class_name] = n
#             total += n
#     return counts, total

# def model_summary_to_string(model):
#     stream = io.StringIO()
#     model.summary(print_fn=lambda x: stream.write(x + '\n'))
#     s = stream.getvalue()
#     stream.close()
#     return s

# def get_env_info():
#     info = {
#         "timestamp": datetime.datetime.now().isoformat(),
#         "python_version": platform.python_version(),
#         "platform": platform.platform(),
#         "tensorflow_version": tf.__version__
#     }
#     try:
#         gpus = tf.config.list_physical_devices('GPU')
#         info['gpus'] = [str(g) for g in gpus]
#     except Exception as e:
#         info['gpus'] = f"error listing GPUs: {e}"
#     return info

# # =========================
# # Load datasets (No splitting)
# # =========================
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TRAIN_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     color_mode="rgb"  # still RGB here, we'll convert later
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     VAL_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     color_mode="rgb"
# )

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     TEST_DIR,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     color_mode="rgb"
# )

# class_names = train_ds.class_names
# print("Detected Classes:", class_names)

# # =========================
# # Preprocessing: RGB → Grayscale → Normalize
# # =========================
# def to_grayscale_and_normalize(image, label):
#     image = tf.image.rgb_to_grayscale(image)   # (H, W, 1)
#     image = tf.cast(image, tf.float32) / 255.0 # Normalize to [0,1]
#     return image, label

# train_ds = train_ds.map(to_grayscale_and_normalize)
# val_ds = val_ds.map(to_grayscale_and_normalize)
# test_ds = test_ds.map(to_grayscale_and_normalize)

# # =========================
# # Plot sample grayscale images (visual check)
# # =========================
# plt.figure(figsize=(10, 8))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().squeeze(), cmap='gray')
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.tight_layout()
# plt.show()

# # =========================
# # Model
# # =========================
# model = models.Sequential([
#     layers.Input(shape=(256, 256, 1)),

#     layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(2, 2),

#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(2, 2),

#     layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(2, 2),

#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(len(class_names), activation='softmax')
# ])

# # =========================
# # Compile & Train
# # =========================
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# # Record training start time
# train_start_time = datetime.datetime.now().isoformat()

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=EPOCHS
# )

# # Record training end time
# train_end_time = datetime.datetime.now().isoformat()

# # =========================
# # Evaluate on Test set
# # =========================
# test_loss, test_acc = model.evaluate(test_ds)
# print(f"Test Accuracy: {test_acc:.4f}")

# # =========================
# # Generate process report (JSON)
# # =========================
# report = {}

# # timestamps
# report['training_start_time'] = train_start_time
# report['training_end_time'] = train_end_time
# report['report_generated_time'] = datetime.datetime.now().isoformat()

# # environment info
# report['environment'] = get_env_info()

# # dataset stats: per split counts and sizes
# splits = {}
# for split_name, split_dir in [("Train", TRAIN_DIR), ("Validation", VAL_DIR), ("Test", TEST_DIR)]:
#     counts_by_class, total = count_files_by_class(split_dir)
#     size_bytes = folder_size_bytes(split_dir)
#     splits[split_name] = {
#         "path": split_dir,
#         "total_images": total,
#         "folder_size_bytes": size_bytes,
#         "folder_size_readable": sizeof_fmt(size_bytes),
#         "per_class_counts": counts_by_class
#     }
# report['dataset'] = splits

# # class names
# report['class_names'] = class_names

# # model summary (text)
# report['model_summary_text'] = model_summary_to_string(model)

# # training history (loss/accuracy per epoch)
# # history.history is typically a dict with keys like 'loss','accuracy','val_loss','val_accuracy'
# report['training_history'] = history.history if hasattr(history, 'history') else {}

# # final test metrics
# report['test'] = {
#     "test_loss": float(test_loss) if test_loss is not None else None,
#     "test_accuracy": float(test_acc) if test_acc is not None else None
# }

# # Save JSON report
# try:
#     with open(REPORT_PATH, 'w', encoding='utf-8') as f:
#         json.dump(report, f, indent=2)
#     print(f"\nProcess report saved to: {REPORT_PATH}")
# except Exception as e:
#     print(f"Error saving report to {REPORT_PATH}: {e}")

