# AI-Based Garbage Sorting Bot

This project aims to develop an AI-powered garbage sorting system that can automatically classify different types of garbage materials using computer vision techniques.

## Table of Contents
- [Introduction](#introduction)
- [Hardware Setup](#hardware-setup)
  - [3D Printed Components](#3d-printed-components)
  - [Parts to be Bought](#parts-to-be-bought)
- [Dataset](#dataset)
- [Software Implementation](#software-implementation)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation)
  - [Real-time Object Detection](#real-time-object-detection)
- [Garbage Sorting Mechanism](#garbage-sorting-mechanism)
- [Results and Accuracy](#results-and-accuracy)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Proper waste management is a critical issue, and accurate garbage classification is the first step towards efficient recycling and disposal. This project aims to create an AI-based garbage sorting system that can automatically recognize and classify different types of garbage materials, such as plastic, paper, metal, and organic waste.

## Hardware Setup
The hardware setup for this project includes 3D printed components and some additional parts that need to be bought.

### 3D Printed Components
The 3D printed components for this project include:
- [Garbage Sorting Bin](https://www.thingiverse.com/thing:1832591)
- [Servo Motor Mount](https://www.thingiverse.com/thing:2920541)
- [Camera Mount](https://www.thingiverse.com/thing:3430866)

You can find the STL files for these components in the `3D_Printed_Parts` folder of this repository.

### Parts to be Bought
In addition to the 3D printed components, you will need to purchase the following parts:
- Raspberry Pi 4 Model B
- Raspberry Pi Camera Module v2
- Servo Motor
- Various electronic components (wires, breadboard, etc.)

## Dataset
The dataset used for this project is the [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) dataset from Kaggle. This dataset contains images of different types of garbage materials, including plastic, paper, metal, and organic waste.

## Software Implementation
The software implementation for this project is done using Python and the TensorFlow deep learning library.

### Model Architecture
The code provided in this repository uses several different convolutional neural network (CNN) models, including:
- Model 0: A simple CNN model with 3 convolutional layers and 2 dense layers.
- Model 1: A CNN model with 3 convolutional layers, 2 dense layers, and an additional dense layer.
- Model 2: A deeper CNN model with 4 convolutional layers, 2 dense layers, and dropout.
- Model 3: A CNN model with data augmentation, 3 convolutional layers, and 2 dense layers.
- Model 4: A deeper CNN model with data augmentation, 4 convolutional layers, and 2 dense layers.
- Model 5: A CNN model with data augmentation, 4 convolutional layers, 2 dense layers, and dropout.

### Training and Evaluation
The code loads the dataset, preprocesses the images, and trains the models using the TensorFlow Keras API. It also includes functions to plot the training and validation loss and accuracy, as well as to evaluate the models on a separate test dataset.

### Real-time Object Detection
The modified code provided in this README includes a `detect_objects_from_webcam` function that uses the trained model to perform real-time object detection on the video feed from the webcam. This allows the system to classify the garbage materials in real-time.

## Garbage Sorting Mechanism
The conveyor belt system is powered by a 12V DC motor operating at 100 RPM, providing the necessary rotational force to drive the belt. The system utilizes a 280 mm GT2 timing belt paired with a 20-tooth pulley, ensuring precise, synchronized movement and preventing any slippage during operation. To minimize friction and enhance efficiency, ball bearings are incorporated, allowing the conveyor to run smoothly while reducing wear and tear on the components.

The rotational motion from the motor is effectively transferred to the pulley, which then converts it into linear motion, driving the conveyor belt forward. This setup is well-suited for applications that require moderate speed and precise control, such as material handling or light manufacturing processes.

To enable automated waste segregation, a camera mount is positioned above the conveyor belt. This camera captures images of the items moving along the belt and, using computer vision algorithms, identifies the type of waste. Based on the waste classification, a further rotary disk mechanism is integrated at the end of the conveyor belt. This rotary disk can rotate and divert the waste items into separate collection bins, segregating the waste into different categories (e.g., plastic, metal, organic) for efficient recycling and disposal.

Proper tensioning of the timing belt is crucial, as it ensures optimal performance and prevents issues like belt slippage or misalignment. The combination of these components results in a reliable, durable, and efficient conveyor system, capable of meeting the demands of various automated waste segregation processes.

## Results and Accuracy
The accuracy of the different models is evaluated on the test dataset, and the best performing model (Model 5) achieves an average accuracy of **98%**. This high accuracy demonstrates the effectiveness of the chosen model architecture and the quality of the dataset.

## Usage
To use this project, you will need to:
1. 3D print the required components.
2. Assemble the hardware setup, including the Raspberry Pi, camera, servo motor, and rotary disk mechanism.
3. Install the necessary software dependencies, including Python, TensorFlow, and OpenCV.
4. Run the provided Python script to train the model and start the real-time object detection and sorting process.

## Future Improvements
Some potential future improvements for this project include:
- Optimizing the mechanical sorting mechanism for faster and more efficient garbage segregation.
- Expanding the dataset to include a wider range of garbage materials.
- Exploring more advanced deep learning architectures for improved classification accuracy.
- Developing a user-friendly interface for the system.
- Integrating the system with smart waste management solutions for comprehensive waste handling.

## Contributing
Contributions to this project are welcome. If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

This project is based on the code from the Kaggle notebook "Material Classifier - TensorFlow CNN" by Omar El Ganainy: [https://www.kaggle.com/code/omarelg/material-classifier-tensorflow-cnn]

The original code has been modified and extended to include additional features, such as 3D printed components, real-time object detection, and the garbage sorting mechanism.
