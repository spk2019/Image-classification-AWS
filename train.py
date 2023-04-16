#import all libraries
from pathlib import Path
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

from utils import SaveBestModel
from utils import save_plots
from model import build_model

import matplotlib.pyplot as plt
plt.style.use("ggplot")


# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=20,
    help='number of epochs to train our network for')
args = vars(parser.parse_args())


# data constants
epochs = args['epochs']
batch_size= 64

######################################################################
####################### Train and test Transforms   ##################
######################################################################

transform_train = transforms.Compose([transforms.Resize(299),
                                        transforms.CenterCrop(299),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

transform_test = transforms.Compose([transforms.Resize(299),
                                        transforms.CenterCrop(299),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

######################################################################
####################### Prepare the Dataset   ########################
######################################################################

data_path = Path("data/Cars")
train_dir = data_path / "train"
test_dir = data_path / "test"
train_dir

train_data = datasets.ImageFolder(root=train_dir,transform = transform_train,target_transform=None)
test_data = datasets.ImageFolder(root=test_dir,transform=transform_test)

trainloader = torch.utils.data.DataLoader(train_data,batch_size=batch_size)
testloader = torch.utils.data.DataLoader(test_data,batch_size=batch_size)

#########################################################################
####################### Loading Pretrained Model ########################
#########################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("GPU Available : ",torch.cuda.get_device_name())


#building the model
model = build_model(pretrained=True,num_classes=2).to(device)
loss_function = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.Adam(model.parameters(),lr=0.01)
# initialize SaveBestModel class
save_best_model = SaveBestModel()


#########################################################################
####################### The Training Script #############################
#########################################################################

def train(dataloader,model,loss_function,optimizer):
    size = len(dataloader.dataset)
    acc = 0
    for batch , (x,y) in enumerate(dataloader):
        x,y = x.to(device), y.to(device)
        outputs = model(x)
        loss = loss_function(outputs,y)
        _, predicted = torch.max(outputs.data, 1)
        acc += (predicted == y).sum().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0 :
            print("Train loss for this epoch : {:.4f}   [{}/{}]".format(loss.item(),batch * len(x),size))  # print when batch = 100 , each batch contains 64 points , so total 6400
    train_accuracy = acc / size
    return loss.item(),train_accuracy      

def test(dataloader,model):
    size = len(dataloader.dataset)
    model.eval()
    acc = 0
    with torch.no_grad():
        
        for batch , (x,y) in enumerate(dataloader):
            x,y = x.to(device), y.to(device)
            outputs = model(x)
            loss = loss_function(outputs,y)
            _, predicted = torch.max(outputs.data, 1)
        
            acc += (predicted == y).sum().item()
            
        test_accuracy =  acc / size  
        print(f"Test accuracy for this epoch : {test_accuracy:.2f}")
    return loss.item(),test_accuracy




#########################################################################
####################### Training Model ##################################
#########################################################################


test_accuracy = []
train_accuracy= []
test_loss=[]
train_loss=[]
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}\n-------------------------------")
    train_loss_epoch,train_accuracy_epoch = train(trainloader, model, loss_function, optimizer)
    test_loss_epoch,test_accuracy_epoch = test(testloader, model)
    
    train_loss.append(train_loss_epoch)
    test_loss.append(test_loss_epoch)
    train_accuracy.append(train_accuracy_epoch)
    test_accuracy.append(test_accuracy_epoch)

    
    save_best_model(
        test_loss_epoch, epoch, model, optimizer, loss_function
    )
    

save_plots(train_accuracy, test_accuracy, train_loss, test_loss)


