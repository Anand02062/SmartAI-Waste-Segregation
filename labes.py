"""
train_tiny_model.py
Trains a SMALL custom CNN suitable for actually fitting on an ESP32
(target: ~50-150KB after int8 quantization). MobileNetV2 is too big.

Dataset folder structure expected:
    dataset/
        plastic/  img1.jpg ...
        metal/    img1.jpg ...
        paper/    img1.jpg ...

Run:
    pip install tensorflow pillow numpy
    python train_tiny_model.py
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
IMG_SIZE = (96, 96)     # small on purpose -- keeps activations tiny for ESP32 RAM
BATCH_SIZE = 16
EPOCHS = 25
CLASSES = ["glass", "metal", "paper", "plastic", "trash"]  # alphabetical order -- must match folder sort order


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode="rgb",
    )

    augment = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ])
    normalize = layers.Rescaling(1.0 / 255)  # simple [0,1] normalization, easy to replicate on ESP32

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_tiny_model(num_classes):
    """A small CNN from scratch -- a handful of conv layers, few filters."""
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),

        layers.Conv2D(8, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),          # 96 -> 48

        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),          # 48 -> 24

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),          # 24 -> 12

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),          # 12 -> 6

        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def representative_dataset_gen(train_ds):
    """Required for full int8 quantization -- feeds sample images so the
    converter can calibrate the int8 ranges."""
    for images, _ in train_ds.take(50):
        for i in range(images.shape[0]):
            yield [np.expand_dims(images[i].numpy(), axis=0)]


def save_dataset_samples(samples_per_class=2):
    """Save a montage containing sample images for every waste class."""
    image_paths = []
    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        files = sorted([
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if not files:
            print(f"WARNING: no images found in {class_dir}")
            continue

        image_paths.append((class_name, files[:samples_per_class]))

    if not image_paths:
        raise RuntimeError("No dataset images were found")

    thumb_width, thumb_height = 180, 180
    canvas_width = thumb_width * samples_per_class
    canvas_height = (thumb_height + 30) * len(image_paths)
    montage = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(montage)

    for row, (class_name, paths) in enumerate(image_paths):
        y = row * (thumb_height + 30)
        draw.text((10, y + 5), class_name.upper(), fill="black")

        for column, path in enumerate(paths):
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((thumb_width, thumb_height))

                x = column * thumb_width + (thumb_width - image.width) // 2
                montage.paste(image, (x, y + 30))

    output_path = Path(__file__).resolve().parent / "dataset_samples.jpg"
    montage.save(output_path)
    print(f"Saved sample montage: {output_path}")

    # Display the generated JPG automatically after every run.
    try:
        montage.show(title="Waste Segregation Dataset Samples")
    except Exception as error:
        print(f"Could not open the preview window: {error}")


def save_individual_dataset_samples(samples_per_class=2):
    """Copy one or two real dataset images from each category into dataset_samples/."""
    output_dir = Path(__file__).resolve().parent / "dataset_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        files = sorted([
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if not files:
            print(f"WARNING: no images found in {class_dir}")
            continue

        for index, source_path in enumerate(files[:samples_per_class], start=1):
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                image.thumbnail((360, 360))
                destination = output_dir / f"{class_name}_{index}.jpg"
                image.save(destination, quality=90)

    print(f"Saved individual examples in: {output_dir}")
    if os.name == "nt":
        os.startfile(output_dir)


def main():
    save_dataset_samples(samples_per_class=2)
    save_individual_dataset_samples(samples_per_class=2)
    train_ds, val_ds = build_datasets()
    model = build_tiny_model(len(CLASSES))
    model.summary()

    print(f"\nApprox parameter count: {model.count_params():,} "
          f"(rough size before quantization: {model.count_params() * 4 / 1024:.0f} KB as float32)\n")

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save("tiny_waste_classifier.keras")

    # --- Full INT8 quantization (required for TFLite Micro on ESP32) ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(train_ds)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    with open("tiny_waste_classifier_int8.tflite", "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nSaved tiny_waste_classifier_int8.tflite -- size: {size_kb:.1f} KB")
    if size_kb > 300:
        print("WARNING: this may be too large for a typical ESP32. "
              "Consider fewer filters/layers or a smaller IMG_SIZE.")

    with open("labels.txt", "w") as f:
        f.write("\n".join(CLASSES))
    print("Saved labels.txt ->", CLASSES)


if __name__ == "__main__":
    main()