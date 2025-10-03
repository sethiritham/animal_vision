import gradio as gr
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from pathlib import Path
import os
from PIL import Image
from model_builder import TinyVGG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ['cat', 'dog', 'elephant', 'horse', 'lion']
EXAMPLES_PATH = Path("data/animals/examples")


def load_models_and_transforms():

    tiny_vgg_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
     
    tiny_vgg = TinyVGG(input_shape=3,
                       output_shape=len(CLASS_NAMES),
                       hidden_units=10)
    
    tiny_vgg.load_state_dict(
        torch.load(f=Path("models/tiny_vgg_model.pth"), map_location=torch.device(DEVICE))
    )
    tiny_vgg.to(DEVICE)


    efficientnet_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    efficientnet_b0 = torchvision.models.efficientnet_b0(weights=None)
    efficientnet_b0.classifier[1] = nn.Linear(in_features=1280, out_features=len(CLASS_NAMES))
    
    efficientnet_b0.load_state_dict(
        torch.load(f=Path("models/efficientnet_b0.pth"), map_location=torch.device(DEVICE))
    )
    efficientnet_b0.to(DEVICE)

    models = {"TinyVGG": tiny_vgg, "EfficientNet-B0": efficientnet_b0}
    model_transforms = {"TinyVGG": tiny_vgg_transform, "EfficientNet-B0": efficientnet_transform}
    
    return models, model_transforms

MODELS, MODEL_TRANSFORMS = load_models_and_transforms()


def predict(image: Image.Image, model_name: str) -> dict[str, float]:

    model = MODELS[model_name]
    transform = MODEL_TRANSFORMS[model_name]


    model.eval()
    with torch.inference_mode():
        transformed_image = transform(image).unsqueeze(dim=0)
        pred_logits = model(transformed_image.to(DEVICE))
    
    pred_probs = torch.softmax(pred_logits, dim=1)
    

    pred_labels_and_probs = {CLASS_NAMES[i]: float(pred_probs[0][i]) for i in range(len(CLASS_NAMES))}
    
    return pred_labels_and_probs


# --- Gradio Interface ---

title = "Animal Vision 🐾"
description = "Compare two models (TinyVGG and EfficientNet-B0) classifying images of various animals."

example_list = [[str(EXAMPLES_PATH / example), "TinyVGG"] for example in os.listdir(EXAMPLES_PATH)]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload an Animal Image"),
        gr.Dropdown(choices=list(MODELS.keys()), value="TinyVGG", label="Select a Model")
    ],
    outputs=gr.Label(num_top_classes=len(CLASS_NAMES), label="Predictions"),
    title=title,
    description=description,
    examples=example_list,
    flagging_mode="never" 
)

demo.launch(share= True)