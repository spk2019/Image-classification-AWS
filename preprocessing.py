from model import build_model
from PIL import Image
import torchvision.transforms as transforms
import torch
import io
import numpy as np



    
def transform_image(image):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
    return my_transforms(image).unsqueeze(0)

def predict_result(tensor):
    model = torch.load('artifacts/model.pkl',map_location='cpu')
    output = model(tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    probs = np.around(probs.detach().cpu().numpy(), 2)
    result = np.argmax(probs[0])
    return result 