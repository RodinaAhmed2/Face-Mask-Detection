import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#تحميل_البيانات

data_dir = "C:/Users/Nada/Desktop/data"
categories = ["with_mask", "without_mask"]
valid_extensions = ['.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tiff']

data = []
labels = []

for category in categories:
    folder_path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    count = 0

    for root, dirs, files in os.walk(folder_path):
        for img_name in files:
            if any(img_name.lower().endswith(ext) for ext in valid_extensions):
                img_path = os.path.join(root, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        data.append(img)
                        labels.append(class_num)
                        count += 1
                except Exception as e:
                    print(f"Error in image: {img_path} - {e}")

    print(f"Loaded {count} images from '{category}'")

if len(data) == 0 or len(labels) == 0:
    print("No images were loaded. Check the path and folders.")
    exit()

data = np.array(data) / 255.0
labels = to_categorical(labels)

print(f"\nTotal loaded images: {len(data)}")

#تقسيم_البيانات

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#بناء_النموذج

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#تدريب_النموذج

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

#نتائج_التدريب

history_df = pd.DataFrame(history.history)
print("\nAccuracy table for each training epoch:")
print(history_df[['accuracy', 'val_accuracy']])

print(f"\nFinal training accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#التقييم

y_true = np.argmax(y_test, axis=1)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Without Mask', 'With Mask'],
            yticklabels=['Without Mask', 'With Mask'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Without Mask', 'With Mask']))

#التنبؤ_بصورة_من_الجهاز

def predict_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("Image not loaded. Check the path.")
            return

        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        label = categories[class_index]
        print(f"\nResult: {label.upper()} (Confidence: {confidence * 100:.2f}%)")

    except Exception as e:
        print(f"Error during prediction: {e}")

#اختيار_صورة

Tk().withdraw()
image_path = askopenfilename(title="Choose an image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.jfif")])
if image_path:
    print(f"Image selected: {image_path}")
    predict_image(image_path)
else:
    print("No image was selected.")
