import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load CSV dataset
csv_path = "gender_detection.csv"  
image_size = (128, 128)  

data = pd.read_csv(csv_path, header=None, names=["image_path", "label", "set"])

label_map = {"man": 0, "woman": 1}
data["label"] = data["label"].map(label_map)

def load_and_preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=image_size)
        img_array = img_to_array(img) / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

X = []
y = []
for _, row in data.iterrows():
    img_array = load_and_preprocess_image(row["image_path"])
    if img_array is not None:
        X.append(img_array)
        y.append(row["label"])

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

model.save("gender_classification_model.h5")

print("✅ Training complete! Model saved as 'gender_classification_model.h5'")
