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
from skimage.metrics import structural_similarity as ssim

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


df=pd.read_csv("tracksfull10000.csv")
df=pd.DataFrame(df) #

tensor_list = [parse_string_to_tensor(data_string) for data_string in df['Matrix']]
X_train_tensors = torch.stack(tensor_list)
#print(X_train_tensors[0])
X_train_matrices = X_train_tensors.unsqueeze(1)
#print(X_train_matrices[0])
batch_size = 32
dataset = CustomDataset(X_train_matrices)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


df2=pd.read_csv("tracksfullnormal10000.csv")
df2=pd.DataFrame(df2)

tensor_list2 = [parse_string_to_tensor(data_string) for data_string in df2['Matrix']]
X_test_tensors = torch.stack(tensor_list2)
X_test_matrices = X_test_tensors.unsqueeze(1)
dataset2 = CustomDataset(X_test_matrices)
test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

df3=pd.read_csv("tracksfull1000.csv")
df3=pd.DataFrame(df3)


tensor_list3 = [parse_string_to_tensor(data_string) for data_string in df3['Matrix']]

# Stack the tensors to create a batch of input data
X_test_tensors2 = torch.stack(tensor_list3)

# Add channel dimension to the tensors (assuming they represent grayscale images)
X_test_matrices2 = X_test_tensors2.unsqueeze(1)
dataset3 = CustomDataset(X_test_matrices2)
validation_loader = DataLoader(dataset3, batch_size=batch_size, shuffle=True)

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 48, 3, padding=1)  #1,48
        self.conv2 = nn.Conv2d(48, 4, 3, padding=1) #48,12
   
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(4, 48, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(48, 1, 2, stride=2)



    def forward(self, x):
        ## encode ##
    
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
     
        ## decode ##
 
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
      
        return x
#Training
model = ConvAutoencoder()
print(model)

# specify loss function
criterion = nn.BCELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 20

epoch_list=[]
train_loss_list=[]

val_losses=[]
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
        #binary_output = (outputs > 0.5).float() 
        # update running training loss
        train_loss += loss.item()*images.size(0)
        
      
    train_loss = train_loss/len(train_loader)
 
    epoch_list.append(epoch)
    train_loss_list.append(train_loss)
  
    

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))
    
    # Validation
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    for data in validation_loader:
        images = data
        outputs = model(images)
        loss = criterion(outputs, images)  # Reconstruction loss
       
        val_loss += loss.item()*images.size(0)
        
    val_loss = val_loss / len(validation_loader)
    val_losses.append(val_loss)
    

   
plt.plot(epoch_list,train_loss_list,label='Training Loss')
plt.plot( epoch_list,val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves BCE')
plt.legend()
plt.show()

#Testing


avg_test_loss_list = []
# Training loop
for epoch in range(1, n_epochs+1):
    # Your training code here

    # Testing
    test_loss = 0.0
    test_loss2 = 0.0
    num_batches = 0
    with torch.no_grad():
        for data in test_loader:
            images = data
            outputs = model(images)
            loss = criterion(outputs, images)
          
            test_loss += loss.item()
       
            num_batches += 1

    # Calculate average test loss for the current epoch
    
    avg_test_loss = test_loss / num_batches
    avg_test_loss_list.append(avg_test_loss)  # Append the average test loss to the list
    

    # Print or log the average test loss for monitoring
    print('Epoch: {} \tAverage Test Loss: {:.6f}'.format(epoch, avg_test_loss))

# Plot the average test loss
plt.plot(epoch_list,avg_test_loss_list,label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Test Loss')
plt.title('Average Test Loss BCE')
plt.legend()
plt.show()



test_loss = 0.0
test_loss2 = 0.0
num_batches = 0
reconstruction_errors = []
reconstruction_errors2 = []
with torch.no_grad():
    for data in test_loader:
        images = data
        outputs = model(images)
        loss = criterion(outputs, images)
        
        test_loss += loss.item()
        
        reconstruction_errors.append(loss.item()*images.size(0))
        
        num_batches += 1
        
    
# Calculate average test loss
avg_test_loss = test_loss / num_batches
print("Average Test Loss:", avg_test_loss)

# Plot histogram of reconstruction errors
plt.hist(reconstruction_errors, bins=20, density=True)

plt.xlabel('Reconstruction Error')
plt.ylabel('Density')
plt.title('Histogram of Reconstruction Errors BCE')
plt.show()

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i].squeeze().detach().numpy(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, 5, i + 6)
    plt.imshow(outputs[i].squeeze().detach().numpy(), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')
plt.show()

