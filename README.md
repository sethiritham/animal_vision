---
title: Animal Vision
emoji: 🐾
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# 🐾 Animal Vision: Multi-Class Image Classification

This Hugging Face Space is a live demo of a multi-class image classification model built with PyTorch. The model is trained to classify images into five categories: **dogs**, **cats**, **elephants**, **horses**, and **lions**.

This project follows a modular programming structure, making the code reusable and easy to understand. It also incorporates TensorBoard for experiment tracking and visualization.

## How to Use This Space

1.  Upload an image of a dog, cat, elephant, horse, or lion using the interface.
2.  Click the "Submit" button.
3.  The model will predict the animal in the image and display the result.

## Features

* **Modular Codebase:** The project is broken down into separate Python scripts for data setup, model building, and training logic.
    * `data_setup.py`: Creates `DataLoader`s for the image data.
    * `model_builder.py`: Defines the CNN architecture (TinyVGG).
    * `engine.py`: Contains reusable functions for training and testing loops.
    * `train.py`: The main script to run the entire training process.
    * `utils.py`: Contains helper functions, such as for saving the model.
* **Experiment Tracking:** Integrated with **TensorBoard** to log and visualize metrics like loss and accuracy.
* **Custom Data:** The model is trained on a custom dataset of animal images.

## Directory Structure

The project is organized as follows:

├── data/
│   └── animals/
│       ├── train/
│       │   ├── dogs/
│       │   ├── cats/
│       │   ├── elephants/
│       │   ├── horses/
│       │   └── lions/
│       └── test/
│           ├── dogs/
│           ├── cats/
│           ├── elephants/
│           ├── horses/
│           └── lions/
├── models/
│   └── tiny_vgg_model.pth
├── runs/
│   └── (TensorBoard log files)
├── going_modular/
│   ├── init.py
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
├── app.py
└── requirements.txt


## How to Run Locally

Follow these instructions to get a copy of the project up and running on your local machine for training or development.

### Prerequisites

* Python 3.8+
* PyTorch
* `torchvision`
* `matplotlib`
* `tensorboard`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sethiritham/pytorch_learning](https://github.com/sethiritham/pytorch_learning) # Replace with your repo URL
    cd pytorch_learning
    ```

2.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare the dataset:** The original training notebook can be used to download the data. Ensure your `data/animals` directory is structured as shown above before training.

### Usage

All scripts should be run from the root project directory.

#### Training the Model

To start training the model, run the `train.py` script as a module:

```bash
python -m going_modular.train
```
This will start the training process, save model checkpoints to the models/ directory, and log results to the runs/ directory for TensorBoard.




### Monitoring with TensorBoard
Open a new terminal and navigate to the project's root directory.
Run the following command:
```bash
tensorboard --logdir=runs
```
Open your web browser and go to http://localhost:6006/.


### Future Improvements
Experiment with different model architectures from torchvision.models (e.g., ResNet, EfficientNet).

Use a larger, more robust dataset.

Implement more advanced data augmentation techniques.

Fine-tune hyperparameters for better performance.

## Acknowledgements
This project was built while following the incredible Zero to Mastery PyTorch for Deep Learning course by Daniel Bourke.
