import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random
from PIL import Image
import PIL
from pathlib import Path
import imghdr
import tensorflow as tf
import shutil


train_parent_path = '/kaggle/working/dataset'
num_classes = 10
try:
    shutil.copytree("/kaggle/input/garbage-classification-v2", train_parent_path)
except FileExistsError:
    print("Dataset already exists")

# function to remove unsupported images
def remove_usupported_images(train_parent_path):
    count=0
    for filepath in Path(train_parent_path).rglob("*"):
        # all images
        if filepath.suffix.lower() in [".png", ".jpg"]:
            img = imghdr.what(filepath)
            # check if image is supported
            if img not in ["jpeg", "png", "bmp", "gif"]:
                os.remove(filepath)
                count+=1
    return count


# function to plot training and validation loss and accuracy
def train_val_loss_accuracy_plot(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Loss')
    plt.show()


 def load_test_dataset(test, target_size=(224, 224)):
    images = []
    labels = []
    class_folders = sorted(os.listdir(test))
    for class_idx, class_folder in enumerate(class_folders):
        class_path = os.path.join(test, class_folder)
        for filename in os.listdir(class_path):
            if filename.lower().endswith((".jpeg")):
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image, target_size)
                images.append(image_resized)
                labels.append(class_idx)
    
    test_images = tf.convert_to_tensor(images, dtype=tf.float32)
    test_labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    return test_images, test_labels


def predict(model, test_images, test_labels, class_names):
    predicted_classes = []
    correct_predictions = 0
    total_samples = len(test_images) 

    for image in test_images:
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)  # Shape becomes (1, 224, 224, 3)

        predictions = model.predict(image)
        score = tf.nn.softmax(predictions[0])
#         print(score)
        predicted_class_idx = np.argmax(score)
        predicted_class_name = class_names[predicted_class_idx]
        predicted_classes.append(predicted_class_name)

    for idx, predicted_class in enumerate(predicted_classes):
        print(f"Sample {idx + 1}: Predicted class is '{predicted_class}', True class is '{class_names[test_labels[idx]]}'")
        if predicted_class == class_names[test_labels[idx]]:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    print("Accuracy is: ",accuracy)


 print(f"Removed {remove_usupported_images(train_parent_path)} images")

train_data = tf.keras.utils.image_dataset_from_directory(
    train_parent_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    train_parent_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=12
)


plt.figure(figsize=(10, 10))
class_names = train_data.class_names
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)



model0 = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])


model1 = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])



model2 = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 4, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes)
])

Data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])
model3 = tf.keras.Sequential([
    Data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])


model4 = tf.keras.Sequential([
    Data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])



model5 = tf.keras.Sequential([
    Data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes)
])


models = [model0, model1, model2, model3, model4, model5]
for model in models:
    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])



epochs = 50


model0_history = model0.fit(
                          train_data,
                          validation_data=val_data,
                          epochs=epochs,
                        )


model1_history = model1.fit(
                          train_data,
                          validation_data=val_data,
                          epochs=epochs,
                        )


model2_history = model2.fit(
                          train_data,
                          validation_data=val_data,
                          epochs=epochs,
                        )

model3_history = model3.fit(train_data, validation_data=val_data, epochs=epochs)

model4_history = model4.fit(train_data, validation_data=val_data, epochs=epochs)
model5_history = model5.fit(
                          train_data,
                          validation_data=val_data,
                          epochs=epochs,
                        )



train_val_loss_accuracy_plot(model0_history, epochs)
train_val_loss_accuracy_plot(model1_history, epochs)
train_val_loss_accuracy_plot(model2_history, epochs)
train_val_loss_accuracy_plot(model3_history, epochs)
train_val_loss_accuracy_plot(model4_history, epochs)


mean_accuracy = np.array(model4_history.history['val_accuracy']).mean()
mean_accuracy

train_val_loss_accuracy_plot(model5_history, epochs)



def detect_objects_from_webcam(model):
    cap = cv2.VideoCapture(0)  # 0 is the default webcam device
    
    while True:
        ret, frame = cap.read()
        
        # Preprocess the frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = tf.expand_dims(image, axis=0)
        
        # Use the model to make predictions
        predictions = model.predict(image)
        score = tf.nn.softmax(predictions[0])
        predicted_class_idx = np.argmax(score)
        predicted_class_name = class_names[predicted_class_idx]
        
        # Display the predicted class on the frame
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Object Detection', frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()




test_data_path = '/kaggle/input/garbagetestdataset/testdataset'
test_images, test_labels = load_test_dataset(test_data_path)


for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(test_images[i].numpy().astype("uint8"))
    plt.title(class_names[test_labels[i]])
    plt.axis("off")


test_accuracy = predict(model0, test_images, test_labels, class_names)
test_accuracy = predict(model3, test_images, test_labels, class_names)
test_accuracy = predict(model4, test_images, test_labels, class_names)
test_accuracy = predict(model5, test_images, test_labels, class_names)



detect_objects_from_webcam(model5)