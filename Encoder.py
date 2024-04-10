#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:23:54 2024

@author: kk423
"""
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import matplotlib.pyplot as plt

#prepare data 

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def parse_string_to_tensor(data_string):
    data_list = eval(data_string)
    return torch.tensor(data_list, dtype=torch.float32)


df=pd.read_csv("tracksfull_data.csv")
df=pd.DataFrame(df) #

tensor_list = [parse_string_to_tensor(data_string) for data_string in df['Matrix']]
X_train_tensors = torch.stack(tensor_list)
#print(X_train_tensors[0])
X_train_matrices = X_train_tensors.unsqueeze(1)
#print(X_train_matrices[0])
batch_size = 32
dataset = CustomDataset(X_train_matrices)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


df2=pd.read_csv("tracksfullnormal_data.csv")
df2=pd.DataFrame(df2)

tensor_list2 = [parse_string_to_tensor(data_string) for data_string in df2['Matrix']]
X_test_tensors = torch.stack(tensor_list2)
X_test_matrices = X_test_tensors.unsqueeze(1)
dataset2 = CustomDataset(X_test_matrices)
test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 48, 2, padding=1)  #1,48
        self.conv2 = nn.Conv2d(48, 12, 2, padding=1) #48,12
   
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(12, 48, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(48, 1, 2, stride=2)



    def forward(self, x):
        ## encode ##
    
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
     
        ## decode ##
 
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
      
        return x
#Training
model = ConvAutoencoder()
print(model)

# specify loss function
criterion = nn.BCELoss()  #nn.MSELoss() 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 30


for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
        images= data
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
            
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

#Testing


test_loss = 0.0
num_batches = 0


with torch.no_grad():
    for data in test_loader:
        images= data
        outputs = model(images)
        loss=criterion(outputs,images)
        test_loss += loss.item()
        num_batches += 1

avg_test_loss = test_loss / num_batches
data_range = torch.max(images) - torch.min(images)
normalized_test_loss = (avg_test_loss / data_range) * 100.0
print("Average Test Loss (Normalized to Percentage): {:.2f}%".format(normalized_test_loss))



test_loss = 0.0
num_batches = 0


with torch.no_grad():
    for images in test_loader:
        outputs = model(images)
        loss = criterion(outputs, images)
        test_loss += loss.item()
        num_batches += 1

# Calculate average test loss
avg_test_loss = test_loss / num_batches
print("Average Test Loss:", avg_test_loss)


for data in test_loader:
    images=data
# get sample outputs
    output = model(images)
# prep images for displayS
    images = images.numpy()
    binary_output = (output > 0.5).float()  # Convert boolean tensor to float tensor (0 or 1)

# Print the binary output
print(images[1])
print(binary_output[1])
    
