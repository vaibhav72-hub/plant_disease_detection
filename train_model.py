"""
Plant Disease Detection Model Training Script
Supports Transfer Learning (MobileNetV2) and Custom CNN on multi-crop datasets.
"""

import os
import sys
import time
import json
import argparse

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def build_transfer_model(num_classes, input_shape=(224, 224, 3)):
    """Build a high-performance transfer learning model using MobileNetV2."""
    print("🧠 Building MobileNetV2 Transfer Learning Architecture...")
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    # Freeze the base feature extraction layers initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def build_custom_cnn(num_classes, input_shape=(224, 224, 3)):
    """Build a lightweight custom Convolutional Neural Network."""
    print("🧱 Building Custom CNN Architecture...")
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model


def train(dataset_path="dataset/PlantVillage",
          architecture="mobilenet",
          epochs=10,
          batch_size=32,
          image_size=(224, 224)):

    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset directory not found at: {dataset_path}")
        print("💡 Please download the dataset first by running:")
        print("   python download_dataset.py\n")
        return

    # Verify class folders
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if len(subdirs) < 2:
        print(f"❌ Found only {len(subdirs)} classes in {dataset_path}. Need at least 2 classes.")
        return

    print("=" * 60)
    print("🌱 PLANT DISEASE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"📁 Dataset Path:    {dataset_path}")
    print(f"🌿 Total Classes:   {len(subdirs)}")
    print(f"⚙️ Architecture:    {architecture.upper()}")
    print(f"🖼️ Image Size:      {image_size}")
    print(f"📦 Batch Size:      {batch_size}")
    print(f"🔄 Epochs:          {epochs}")
    print("=" * 60)

    # -----------------------------
    # Data Augmentation & Generators
    # -----------------------------
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    print("⏳ Loading training and validation datasets...")
    train_data = train_gen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_data = train_gen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    num_classes = train_data.num_classes
    print(f"✅ Found {train_data.samples} training images and {val_data.samples} validation images across {num_classes} classes.")

    # -----------------------------
    # Model Compilation
    # -----------------------------
    if architecture.lower() == "mobilenet":
        model = build_transfer_model(num_classes, input_shape=(image_size[0], image_size[1], 3))
        lr = 0.0005
    else:
        model = build_custom_cnn(num_classes, input_shape=(image_size[0], image_size[1], 3))
        lr = 0.0001

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -----------------------------
    # Callbacks
    # -----------------------------
    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint("model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    # -----------------------------
    # Training Execution
    # -----------------------------
    start_time = time.time()
    print("\n🚀 Commencing model training...")
    
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks
    )

    training_time = time.time() - start_time

    # -----------------------------
    # Save Model and Class Names
    # -----------------------------
    model.save("model.h5")
    class_indices = train_data.class_indices
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

    with open("class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)

    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("💾 Saved weights to:        model.h5")
    print("📋 Saved class mappings to: class_indices.json")
    print(f"⏱️ Total Training Time:     {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Report final accuracy
    val_acc = history.history.get("val_accuracy", [-1])[-1]
    train_acc = history.history.get("accuracy", [-1])[-1]
    print(f"🎯 Final Training Accuracy:   {train_acc * 100:.2f}%")
    print(f"🎯 Final Validation Accuracy: {val_acc * 100:.2f}%")
    print("=" * 60)
    print("👉 Now start your Flask web app using: python app.py\n")


def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Model")
    parser.add_argument("--dataset", type=str, default="dataset/PlantVillage", help="Path to dataset root folder")
    parser.add_argument("--architecture", type=str, default="mobilenet", choices=["mobilenet", "cnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
