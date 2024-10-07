
# Code Analysis
## THESE LIBRARIES **MUST** BE INSTALLED. 
## if the code is opened on VSCODE, press `ctrl + shift + v` to toggle between views for the markdown document
## CHECK THE END OF THE DOCUMENT FOR GITHUB REPOSITORIES AND PAPERS
## Library Installation
```python
%pip install torch torchvision pillow matplotlib opencv-python grad-cam
```
2 addtional libraries are added here: `opencv-python` and `grad-cam`
This command installs the necessary Python libraries using pip. These libraries include:
- `torch`: PyTorch library for deep learning.
- `torchvision`: for computer vision tasks.
- `pillow`: Image processing library.
- `matplotlib`: Plotting library.
- `opencv-python`: OpenCV library for computer vision.
- `grad-cam`: Library for Gradient-weighted Class Activation Mapping.

## Loading and Inspecting the Model Checkpoint
```python
import torch
import torch.nn as nn

checkpoint_path = 'malaria_AI.pth'
checkpoint = torch.load(checkpoint_path)
print("Checkpoint keys:", checkpoint.keys())

if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        print(key)
elif isinstance(checkpoint, torch.nn.Module):
    print(checkpoint)
else:
    print("Unexpected checkpoint format:", type(checkpoint))
```
This code loads a PyTorch model checkpoint and inspects its keys. Depending on the checkpoint format, it prints the keys if it's a dictionary or the model architecture if it's a `torch.nn.Module`.

## Setting Device for Computation
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
```
This code checks if a GPU is available. If yes, it sets the computation device to GPU (`cuda`); otherwise, it sets it to CPU (`cpu`). Even if a GPU isnt available on your computer, the code still going to work.

## Defining the CNN Model
```python
class CNN_1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 289, out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

input_shape = 3
hidden_units = 12
output_shape = 2
model = CNN_1(input_shape, hidden_units, output_shape).to(device)
```
This code defines the Convolutional Neural Network (CNN) model with two convolutional blocks and a classifier block. The model is then instantiated with specific input, hidden, and output dimensions and moved to device. The chosen numbers arent random.

## Loading the Model State Dictionary
```python
model.load_state_dict(torch.load('malaria_AI.pth'))
model.eval()
```
This code loads the model parameters from the checkpoint (Malaria_AI) and sets the model to evaluation mode. Setting the model to `eval()` is very important because it stops the model from updating it parameters or training.

## Preprocessing and Prediction
the code below defines a resizing function, a function to get the image and one other function that passes the the image through the model. all the images end with .png
```python
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def get_random_image(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    random_subfolder = random.choice(subfolders)
    images = [os.path.join(random_subfolder, img) for img in os.listdir(random_subfolder) if img.endswith('.png')]
    return random.choice(images)

def predict(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
    return output
```
Grad-CAM, or Gradient-weighted Class Activation Mapping, allows us to visualize the regions of an image that are important for a Convolutional Neural Network (CNN) in making its predictions. The GitHub repo for Grad-CAM is at the end of the document.
## Generating and Displaying Grad-CAM
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_and_show_gradcam(model, target_layer, image_tensor, predicted_label):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)[0, :]
    original_image = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    original_image = original_image / original_image.max()  # Normalize to [0, 1]
    '''
    the images values are normalized from [0, 255] to [0, 1]
    '''
    visualization = show_cam_on_image(original_image, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.7)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original_image)
    ax[0].axis('off')
    ax[0].set_title(f'{predicted_label} | Original Image')

    ax[1].imshow(visualization)
    ax[1].axis('off')
    ax[1].set_title(f'{predicted_label} | GradCAM')

    plt.show()
```
This function generates and displays a Grad-CAM visualization, highlighting the regions of the input image that the CNN model focuses on for making predictions. The `GradCAM` class is used to compute the Grad-CAM and `show_cam_on_image` to overlay the Grad-CAM on the original image.

## Example Usage
```python
folder_path = 'Test'
random_image_path = get_random_image(folder_path)
image_tensor = preprocess_image(random_image_path)
output = predict(image_tensor, model)
_, predicted = torch.max(output, 1)
class_labels = {0: 'parasite', 1: 'normal'}
predicted_label = class_labels[predicted.item()]
print(f'Predicted class for {random_image_path}: {predicted_label}')

target_layer = model.conv_block_2[-1]
generate_and_show_gradcam(model, target_layer, image_tensor, predicted_label)
```
* `_, predicted = torch.max(output, 1)` --> `_` represents the actual raw outputs of the model while `predicted` returns the argmax of `output`.

* the target layer `model.conv_block_2[-1]` isnt chosen at random also. It refers to the last layer of the Convolutional Neural Network.
And to view succesive images, you dont have to run all the cells again, just the last cell, unlike the former code.

# LINKS TO GITHUB REPOS AND PAPERS
* Grad-CAM
    * Repository --> https://github.com/jacobgil/pytorch-grad-cam
    * Paper --> https://arxiv.org/abs/1610.02391