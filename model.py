import torchvision.models as models
import torch.nn as nn
import torch




def build_model(pretrained, num_classes):

    inception = torchvision.models.inception_v3(pretrained=True)

    #freezing all hidden layers
    for param in inception.parameters():
        param.requires_grad=False

    #Training the last fc layer
    in_features = inception.fc.in_features
    inception.fc = nn.Linear(in_features=in_features,out_features=2)
            
    return inception



