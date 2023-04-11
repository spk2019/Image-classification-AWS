import torchvision.models as models
import torch.nn as nn
import torch




def build_model(pretrained, num_classes):

    resnet = models.resnet50(pretrained=pretrained)

    #freezing all hidden layers
    for param in resnet.parameters():
        param.requires_grad=False

    #Training the last fc layer
    in_features =  resnet.fc.in_features
    resnet.fc = nn.Linear(in_features,num_classes)
            
    return resnet



