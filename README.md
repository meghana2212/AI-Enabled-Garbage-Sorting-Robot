📚 Table of Contents

    Overview

    System Components

        Custom 3D Printed Parts

        Electronics and Hardware

    Dataset Information

    AI Model Implementation

        Neural Network Designs

        Training Workflow

        Live Detection Pipeline

    Mechanical Sorting Process

    Performance and Evaluation

    Getting Started

    Potential Enhancements

    How to Contribute

    License and Attribution

🔍 Overview

Efficient waste separation is key to modern recycling efforts. This project introduces an automated garbage sorting system driven by AI and computer vision that can classify waste into categories like plastic, paper, metal, and organic in real-time.
🛠 System Components
🔧 Custom 3D Printed Parts

To support the hardware design, 3D printed parts are used for mounting and structure:

    Sorting Bin Container

    Servo Holder Assembly

    Camera Bracket

STL files can be found in the 3D_Printed_Parts folder within the project directory.
⚙️ Electronics and Hardware

To assemble the sorting mechanism, the following items are required:

    Raspberry Pi 4 (Model B)

    Raspberry Pi Camera Module v2

    Servo Motor (compatible with Raspberry Pi)

    12V DC Motor (100 RPM)

    GT2 Timing Belt (280 mm) and Pulley (20 teeth)

    Ball Bearings

    Miscellaneous electronics (breadboard, jumpers, resistors, etc.)

🧠 Dataset Information

We use the Garbage Classification V2 Dataset from Kaggle. It contains labeled images for several waste categories including plastic, paper, metal, and organic waste, suitable for training a classification model.
🧩 AI Model Implementation
📐 Neural Network Designs

Several CNN models have been implemented and evaluated:

    Model A: Basic CNN with 3 conv layers and 2 dense layers.

    Model B: Similar to A but with an additional dense layer.

    Model C: Deeper network with 4 conv layers, dropout included.

    Model D: Includes data augmentation, 3 conv layers, and dense layers.

    Model E: Augmented input with 4 conv layers and improved structure.

    Model F: Advanced design using data augmentation and dropout layers for generalization.

🏋️ Training Workflow

Models are trained using TensorFlow’s Keras API. The script handles image preprocessing, model compilation, training, validation, and test evaluation. Graphs for loss and accuracy are generated automatically.
🎥 Live Detection Pipeline

A dedicated function detect_objects_from_webcam allows real-time classification using the webcam feed. Once deployed on the Raspberry Pi, the system can sort waste as it moves along the conveyor belt.
🔄 Mechanical Sorting Process

The sorting mechanism is based on a conveyor system driven by a 12V DC motor (100 RPM) with a GT2 belt and pulley combo. The camera, mounted above the belt, captures images for classification.

After classification:

    A rotary disk mechanism is triggered by a servo motor.

    The disk rotates to direct waste into the correct bin.

Ball bearings reduce friction, and proper tensioning of the belt ensures consistent, reliable operation.
📊 Performance and Evaluation

Each model is evaluated using a held-out test set. The best-performing model (Model F) achieved an average accuracy of 98%, demonstrating strong potential for real-world use.
🚀 Getting Started

To deploy the system:

    Print the required components listed in the 3D print section.

    Assemble the mechanical parts and connect the electronics.

    Install Python packages: TensorFlow, OpenCV, etc.

    Train a model or use the provided pretrained weights.

    Run the detection and sorting script to begin operation.

🔧 Potential Enhancements

Ideas for extending the project:

    Faster and more compact mechanical design.

    Support for more waste categories (e.g., glass, textiles).

    Integration with mobile/web dashboard for monitoring.

    Deployment-ready edge AI using TensorFlow Lite.

    Energy-efficient operation powered by solar panels.



This project is distributed under the MIT License.

Original inspiration and base code from:
Omar El Ganainy's Kaggle notebook - Material Classifier - TensorFlow CNN
Our version includes new models, real-time detection, mechanical integration, and full hardware documentation.
