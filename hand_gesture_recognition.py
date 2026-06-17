import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path = "D:\GITDEMO\SCT_ML_4\leapGestRecog"

images = []
labels = []

gesture_names = {
    "01_palm": 0,
    "02_l": 1,
    "03_fist": 2,
    "04_fist_moved": 3,
    "05_thumb": 4,
    "06_index": 5,
    "07_ok": 6,
    "08_palm_moved": 7,
    "09_c": 8,
    "10_down": 9
}

gesture_labels = [
    "Palm",
    "L",
    "Fist",
    "Fist Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm Moved",
    "C",
    "Down"
]

print("Loading Dataset...")

for subject in os.listdir(dataset_path):

    subject_path = os.path.join(dataset_path, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):

        gesture_path = os.path.join(subject_path, gesture)

        if gesture not in gesture_names:
            continue

        for image_name in os.listdir(gesture_path):

            image_path = os.path.join(
                gesture_path,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.resize(
                image,
                (64, 64)
            )

            images.append(image)

            labels.append(
                gesture_names[gesture]
            )

print("Dataset Loaded Successfully")

X = np.array(images, dtype=np.float32) / 255.0
y = np.array(labels)

plt.figure(figsize=(10, 5))

sns.countplot(
    x=y
)

plt.xticks(
    range(10),
    gesture_labels,
    rotation=45
)

plt.title("Gesture Distribution")
plt.xlabel("Gesture")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "gesture_distribution.png"
)

plt.show()

y = to_categorical(
    y,
    num_classes=10
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(64, 64, 3)
    )
)

model.add(
    MaxPooling2D(
        (2, 2)
    )
)

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(
    MaxPooling2D(
        (2, 2)
    )
)

model.add(
    Flatten()
)

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(
    Dropout(
        0.3
    )
)

model.add(
    Dense(
        10,
        activation="softmax"
    )
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting Model Training...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"]
)

plt.plot(
    history.history["val_accuracy"]
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Training Accuracy",
    "Validation Accuracy"
])

plt.savefig(
    "accuracy_graph.png"
)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"]
)

plt.plot(
    history.history["val_loss"]
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Training Loss",
    "Validation Loss"
])

plt.savefig(
    "loss_graph.png"
)

plt.show()

sample_index = np.random.randint(
    0,
    len(X_test)
)

sample_image = X_test[sample_index]

prediction = model.predict(
    np.expand_dims(
        sample_image,
        axis=0
    ),
    verbose=0
)

predicted_class = np.argmax(
    prediction
)

confidence = np.max(
    prediction
) * 100

plt.figure(figsize=(8, 6))

plt.imshow(
    sample_image
)

plt.axis("off")

plt.figtext(
    0.10,
    0.08,
    f"Predicted Gesture: {gesture_labels[predicted_class]}",
    fontsize=14
)

plt.figtext(
    0.10,
    0.03,
    f"Confidence: {confidence:.2f}%",
    fontsize=14
)

plt.savefig(
    "gesture_prediction.png",
    bbox_inches="tight"
)

plt.show()

model.save(
    "hand_gesture_model.h5"
)

print("\nModel Saved Successfully")