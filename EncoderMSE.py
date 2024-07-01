#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:18:15 2024

@author: kk423
"""

import mlflow
import mlflow.pytorch
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
#from torch.quantization import QuantStub, DeQuantStub
from brevitas.nn import QuantConv2d, QuantReLU, QuantMaxPool2d, QuantConvTranspose2d, QuantSigmoid,QuantIdentity, QuantLinear,QuantHardTanh
from matplotlib.lines import Line2D
from torchinfo import summary
from brevitas.nn import QuantConv2d, QuantReLU
import os

   
def CustomLoss(predictions, targets):
        for image,output in zip(targets,predictions):
            image=torch.sum(image)
            output=torch.sum(output)
            return torch.abs(output-image) 


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


df=pd.read_csv("train_datab4.csv")
df=pd.DataFrame(df) #
 

tensor_list = [parse_string_to_tensor(data_string) for data_string in df['Matrix']]
X_train_tensors = torch.stack(tensor_list)
#print(X_train_tensors[0])
X_train_matrices = X_train_tensors.unsqueeze(1)
#print(X_train_matrices[0])
batch_size = 32# 32
dataset = CustomDataset(X_train_matrices)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


df2=pd.read_csv("test_datab4.csv")
df2=pd.DataFrame(df2)

tensor_list2 = [parse_string_to_tensor(data_string) for data_string in df2['Matrix']]
X_test_tensors = torch.stack(tensor_list2)
X_test_matrices = X_test_tensors.unsqueeze(1)
dataset2 = CustomDataset(X_test_matrices)
test_loader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)

df3=pd.read_csv("validation_datab4.csv")
df3=pd.DataFrame(df3)


tensor_list3 = [parse_string_to_tensor(data_string) for data_string in df3['Matrix']]

# Stack the tensors to create a batch of input data
X_test_tensors2 = torch.stack(tensor_list3)

# Add channel dimension to the tensors (assuming they represent grayscale images)
X_test_matrices2 = X_test_tensors2.unsqueeze(1)
dataset3 = CustomDataset(X_test_matrices2)
validation_loader = DataLoader(dataset3, batch_size=batch_size, shuffle=True)


#testing on specific data
bound = 10
curved_data = df2[df2['Curvature'].abs() <= bound]
straight_data = df2[df2['Curvature'].abs() >= bound]


tensor_list4 = [parse_string_to_tensor(data_string) for data_string in curved_data['Matrix']]

# Stack the tensors to create a batch of input data
curved_tensors = torch.stack(tensor_list4)

# Add channel dimension to the tensors (assuming they represent grayscale images)
curved_matrices = curved_tensors.unsqueeze(1)
dataset4 = CustomDataset(curved_matrices)
curved_loader = DataLoader(dataset4, batch_size=batch_size, shuffle=True)

tensor_list5 = [parse_string_to_tensor(data_string) for data_string in straight_data['Matrix']]

# Stack the tensors to create a batch of input data
straight_tensors = torch.stack(tensor_list5)

# Add channel dimension to the tensors (assuming they represent grayscale images)
straight_matrices = straight_tensors.unsqueeze(1)
dataset5 = CustomDataset(straight_matrices)
straight_loader = DataLoader(dataset5, batch_size=batch_size, shuffle=True)

curved_data2 = df3[df3['Curvature'].abs() <= bound]
straight_data2 = df3[df3['Curvature'].abs() >= bound]


tensor_list6= [parse_string_to_tensor(data_string) for data_string in curved_data2['Matrix']]

# Stack the tensors to create a batch of input data
curved_tensors2 = torch.stack(tensor_list6)

# Add channel dimension to the tensors (assuming they represent grayscale images)
curved_matrices2 = curved_tensors2.unsqueeze(1)
dataset6 = CustomDataset(curved_matrices2)
curved_loader2 = DataLoader(dataset6, batch_size=batch_size, shuffle=True)

tensor_list7 = [parse_string_to_tensor(data_string) for data_string in straight_data2['Matrix']]

# Stack the tensors to create a batch of input data
straight_tensors2 = torch.stack(tensor_list7)

# Add channel dimension to the tensors (assuming they represent grayscale images)
straight_matrices2 = straight_tensors2.unsqueeze(1)
dataset7 = CustomDataset(straight_matrices2)
straight_loader2 = DataLoader(dataset7, batch_size=batch_size, shuffle=True)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1),
            QuantReLU(bit_width=1),
            #nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 24, 3, padding=1),
            QuantReLU(bit_width=1),
            #nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(72,48),
            #nn.ReLU(),
            QuantReLU(bit_width=1),
            nn.Linear(48,24),
            #nn.ReLU(),
            QuantReLU(bit_width=1),
            nn.Linear(24,12),
            QuantReLU(bit_width=1)
            
        )
        
  
        # Decoder layers
        self.decoder = nn.Sequential(
         nn.Linear(12, 24),
         #nn.ReLU(),
         QuantReLU(bit_width=8),
         nn.Linear(24,48),
         #nn.ReLU(),
         QuantReLU(bit_width=8),
         nn.Linear(48,72),  # Reverse the flattening process for the convolutional input
         #nn.ReLU(),
         QuantReLU(bit_width=8),
         nn.Unflatten(1, (24, 1, 3)),  # Reshape back to (4, 7, 7) to match ConvTranspose input
         nn.ConvTranspose2d(24, 48, 2, stride=2),
         #nn.ReLU(),
         QuantReLU(bit_width=8),
         nn.ConvTranspose2d(48, 1, 2, stride=2),
         nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        enc=self.encoder(x)
        encoded=torch.clamp(enc,min=None,max=1)
    
        decoded = self.decoder(encoded)
        
        return decoded, encoded

# When initialzing, it will run __init__() function as above

#print(model)

def train_model(model, train_loader, validation_loader, n_epochs=40, lr=0.001, weight_decay=0.00001):
    criterion = nn.MSELoss()
   # criterion2 = PixelWiseMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    train_loss_list = []
    val_losses = []
    a = 0.0000001
    b = 0
    c = 1 - a - b
    l1_lambda = 0.001

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for data in train_loader:
            images = data
            optimizer.zero_grad()
            outputs = model(images)[0]
            loss1 = criterion(outputs, images)
            #loss2 = criterion2(outputs, images)
            loss3 = CustomLoss(outputs, images)
            loss = c * loss1 + a * loss3 

            L1_term = torch.tensor(0., requires_grad=True)
            for name, weights in model.named_parameters():
                if 'bias' not in name:
                    weights_sum = torch.sum(torch.abs(weights))
                    L1_term = L1_term + weights_sum
            L1_term = L1_term / sum(p.numel() for p in model.parameters() if 'bias' not in name)
            loss = loss - L1_term * l1_lambda
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        train_loss = train_loss / len(train_loader)
        train_loss_list.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in validation_loader:
                images = data
                outputs = model(images)[0]
                loss1 = criterion(outputs, images)
                #loss2 = criterion2(outputs, images)
                loss3 = CustomLoss(outputs, images)
                loss = c * loss1 + a * loss1 
                val_loss += loss.item() * images.size(0)
            val_loss = val_loss / len(validation_loader)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {val_loss:.6f}')
    
    return train_loss_list, val_losses

def evaluate_model(model, test_loader, n_images=16, save_dir='images', session=1):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    #criterion2 = PixelWiseMSELoss()
    a = 0.0000001
    b = 0
    c = 1 - a - b

    with torch.no_grad():
        for data in test_loader:
            images = data
            outputs = model(images)[0]
            outputs2 = model(images)[1]
            loss1 = criterion(outputs, images)
            #loss2 = criterion2(outputs, images)
            loss3 = CustomLoss(outputs, images)
            loss = c * loss1 + a * loss3
            test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print("Average Test Loss:", avg_test_loss)

        # Save images
        fig, axes = plt.subplots(3, n_images, figsize=(n_images + 10, 4))
        for i in range(n_images):
            ax = axes[0, i]
            ax.imshow(images[i].squeeze().detach().numpy(), cmap='gray')
            ax.set_title('Original', fontsize=10)
            ax.axis('off')

            ax = axes[1, i]
            ax.imshow(outputs[i].squeeze().detach().numpy(), cmap='gray')
            ax.set_title('Output', fontsize=10)
            ax.axis('off')

            reshaped_tensor = outputs2.view(n_images, 1, 4, 3)
            ax = axes[2, i]
            ax.imshow(reshaped_tensor[i].squeeze().detach().numpy(), cmap='gray')
            ax.set_title('12-bit Output', fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        image_path = os.path.join(save_dir, f'output_images_session_{session}.png')
        plt.savefig(image_path)
        plt.show()

    return avg_test_loss

def main(n_iterations=5):
    # Load data
   # train_loader, validation_loader, test_loader, straight_loader, curved_loader = load_data()

    for i in range(n_iterations):
        print(f"--- Iteration {i + 1} ---")
        model = ConvAutoencoder()
        train_loss_list, val_losses = train_model(model, train_loader, validation_loader)
        avg_test_loss = evaluate_model(model, test_loader, session=i + 1)

main(n_iterations=5)  # Change the number of iterations as needed