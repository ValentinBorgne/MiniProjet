import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ==============================
# PARTIE 1 : Exploration des données
# ==============================


images_path = "MiniProjet/images"

all_files = os.listdir(images_path)

all_images = [f for f in all_files if f.lower().endswith(".jpg")]

cats = [img for img in all_images if img.startswith("cat")]
dogs = [img for img in all_images if img.startswith("dog")]

print("Chats :", len(cats))
print("Chiens :", len(dogs))


plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    img_name = random.choice(all_images)
    img_path = os.path.join(images_path, img_name)

    img = load_img(img_path)
    plt.imshow(img)

    plt.title("Chat" if img_name.startswith("cat") else "Chien")
    plt.axis("off")

plt.show()


plt.show()

sample_img = random.choice(all_images)
img = load_img(os.path.join(images_path, sample_img))

print("Nom de l'image :", sample_img)
print("Taille de l'image :", img.size)

img = load_img(
    os.path.join(images_path, sample_img),
    target_size=(150, 150)
)

img_array = img_to_array(img)
img_array = img_array / 255.0

print("Shape :", img_array.shape)
print("Min :", img_array.min())
print("Max :", img_array.max())

# ==============================
# PARTIE 2 : Préparation des données
# ==============================


base_dir = "Mini/projetdataset"
subdirs = ["train", "val", "test"]
classes = ["cats", "dogs"]

for subdir in subdirs:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, subdir, cls), exist_ok=True)

random.shuffle(cats)
random.shuffle(dogs)

def split_and_copy(files, class_name):
    n = len(files)
    train_end = int(0.7 * n)
    val_end = int(0.9 * n)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, filenames in splits.items():
        for fname in filenames:
            src = os.path.join(images_path, fname)
            dst = os.path.join(base_dir, split, class_name, fname)
            shutil.copy(src, dst)

split_and_copy(cats, "cats")
split_and_copy(dogs, "dogs")

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "MiniProjet/dataset/train",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    "MiniProjet/dataset/val",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    "MiniProjet/dataset/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary"
)

# ==============================
# PARTIE 3 : Création d'un modèle CNN
# ==============================



model = Sequential([

    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ==============================
# PARTIE 4 : Entrainement du modèle
# ==============================

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)


plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ==============================
# PARTIE 5 : Amelioration du modèle (Dropout)
# ==============================

model_dropout = Sequential([

    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model_dropout.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_dropout = model_dropout.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=5
)
# Accuracy
plt.plot(history.history['val_accuracy'], label='Val accuracy AVANT')
plt.plot(history_dropout.history['val_accuracy'], label='Val accuracy APRÈS')
plt.title("Comparaison Accuracy Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss
plt.plot(history.history['val_loss'], label='Val loss AVANT')
plt.plot(history_dropout.history['val_loss'], label='Val loss APRÈS')
plt.title("Comparaison Loss Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ==============================
# PARTIE 6 : Test du modele
# ==============================

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

test_dir = "MiniProjet/test_images"
test_images = os.listdir(test_dir)

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)

    img = preprocess_image(img_path)
    prediction = model.predict(img)[0][0]

    plt.imshow(load_img(img_path))
    plt.axis("off")

    if prediction < 0.5:
        label = "Chat"
        proba = (1 - prediction) * 100
    else:
        label = "Chien"
        proba = prediction * 100

    plt.title(f"{label} - Probabilité : {proba:.2f}%")
    plt.show()