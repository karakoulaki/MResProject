#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:50:31 2024

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
import torch.nn.init as init
   
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
class PixelWiseBCELoss(nn.Module):
    def __init__(self):
        super(PixelWiseBCELoss, self).__init__()

    def forward(self, input, target):
        #assert input.shape == target.shape, "Input and target must have the same shape."
        for i,j in zip(input,target):
            
        # Apply sigmoid to ensure input values are in the range [0, 1]
        #input = torch.sigmoid(input)
        
        # Calculate the BCE for each pixel
            pixel_bce = F.binary_cross_entropy(i ,j, reduction='none')
        
        # Return the pixel-wise MSE as a tensor
        return pixel_bce

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


        
# Define the weight initialization function
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight,  mode='fan_in',nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight,  mode='fan_in',nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
          
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)       
            

# Create the model
model = ConvAutoencoder()

# Apply the custom weight initialization
model.apply(initialize_weights)
#print(model)

# specify loss function
criterion = nn.MSELoss()
criterion2 = nn.BCELoss() 
#criterion2  = PixelWiseBCELoss()

a =0.0000001 
b = 0.6
c = 1-a-b
weight_decay=0.00001
lr=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay) #0.00001
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


nweights = 0
for name,weights in model.named_parameters():
    if 'bias' not in name:
        nweights = nweights + weights.numel()
    
print(f'Total number of weights in the model = {nweights}')
#scheduler = torch.ReduceLROnPlateau(optimizer, 'min')
# Extract weights and biases
# Create a dictionary mapping internal parameter names to descriptive names
name_mapping = {
    'encoder.0.weight': 'Encoder Conv Layer 1 Weights',
    'encoder.0.bias': 'Encoder Conv Layer 1 Biases',
    'encoder.3.weight': 'Encoder Conv Layer 2 Weights',
    'encoder.3.bias': 'Encoder Conv Layer 2 Biases',
    'encoder.7.weight': 'Encoder Linear 1 Weights',
    'encoder.7.bias': 'Encoder Linear Layer 1 Biases',
    'encoder.9.weight': 'Encoder Linear Layer 2 Weights',
    'encoder.9.bias': 'Encoder Linear Layer 2 Biases',
    'encoder.11.weight': 'Encoder Linear Layer 3 Weights',
    'encoder.11.bias': 'Encoder Linear Layer 3 Biases',
    'decoder.0.weight': 'Decoder ConvT Layer 1 Weights',
    'decoder.0.bias': 'Decoder ConvT Layer 1 Biases',
    'decoder.2.weight': 'Decoder ConvT Layer 2 Weights',
    'decoder.2.bias': 'Decoder ConvT Layer 2 Biases',
    'decoder.4.weight': 'Decoder Linear Layer 1 Weights',
    'decoder.4.bias': 'Decoder Linear Layer 1 Biases',
    'decoder.7.weight': 'Decoder Linear Layer 2 Weights',
    'decoder.7.bias': 'Decoder Linear Layer 2 Biases',
    'decoder.9.weight': 'Decoder Linear Layer 3 Weights',
    'decoder.9.bias': 'Decoder Linear Layer 3 Biases'
}

# Define colors for each layer
layer_colors = ['blue', 'orange', 'green', 'red','black','yellow','purple','lightgreen','grey','deepskyblue']

# Extract weights and biases by layer with more descriptive names
weights_by_layer = []
biases_by_layer = []
weight_names = []
bias_names = []

for name, param in model.named_parameters():
    if 'weight' in name:
        weights_by_layer.append(param.data.numpy().flatten())
        weight_names.append(name_mapping[name])
    elif 'bias' in name:
        biases_by_layer.append(param.data.numpy().flatten())
        bias_names.append(name_mapping[name])
# Plot histograms for weights"""
plt.figure(figsize=(12, 6))


# Weights histogram
plt.subplot(1, 2, 1)
for i, (weights, name) in enumerate(zip(weights_by_layer, weight_names)):
    plt.hist(weights, bins=50, color=layer_colors[i % len(layer_colors)], alpha=0.5, label=name)
plt.title('Weights Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# Biases histogram
plt.subplot(1, 2, 2)
for i, (biases, name) in enumerate(zip(biases_by_layer, bias_names)):
    plt.hist(biases, bins=50, color=layer_colors[i % len(layer_colors)], alpha=0.5, label=name)
plt.title('Biases Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()



n_epochs = 30
l1_lambda = 0.001
losses = np.zeros(n_epochs+1)

epoch_list=[]
train_loss_list=[]

val_losses=[]
val_losses2=[]
val_losses3=[]

    


for epoch in range(1, n_epochs+1):
        
        model.train()
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
            outputs = model(images)[0]
            # calculate the loss
            
            loss1 = criterion(outputs,images)
            
            loss2 = criterion2(outputs,images)
            
            loss3 = CustomLoss(outputs,images)
            
            loss= c*loss1+ a*loss3 +b*loss2 
            
            
            L1_term = torch.tensor(0., requires_grad=True)
            for name, weights in model.named_parameters():
               if 'bias' not in name:
                    weights_sum = torch.sum(torch.abs(weights))
                    L1_term = L1_term + weights_sum
            L1_term = L1_term / nweights
            #loss_l1=0
            #for parm in model.parameters():
             #   loss_l1+=torch.sum(torch.abs(parm))
            #loss+=lan_l1*loss_l1
                        
            loss = loss - L1_term * l1_lambda
                        #losses[epoch] = loss.detach().item()
       
            optimizer.zero_grad()
            loss.backward()
    
            # perform a single optimization step (parameter update)
            optimizer.step()
   
            train_loss += loss.item()*images.size(0)
        
      
        
        
      
        train_loss = train_loss/len(train_loader)
       
        #mlflow.log_metric("accuracy", f"{accuracy:2f}", step=epoch)
        
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        
        

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))
    
    # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for data in validation_loader:
                images = data
                outputs = model(images)[0]
                loss1 = criterion(outputs,images)
                
                loss2 = criterion2(outputs,images)
                
                loss3 = CustomLoss(outputs,images)
                
                loss= c*loss1+  a*loss1 +b*loss2
                val_loss += loss.item()*images.size(0)
                
            val_loss = val_loss / len(validation_loader)
            val_losses.append(val_loss)
            mlflow.log_metric("eval_loss", val_loss, epoch)
            scheduler.step(val_loss)   
            model.eval() 
            val_loss2 = 0.0
        with torch.no_grad():
            for data in straight_loader2:
                images = data
                outputs = model(images)[0]
                loss1 = criterion(outputs,images)
                
                loss2 = criterion2(outputs,images)
                loss3 = CustomLoss(outputs,images)
         
                loss= c*loss1 + a*loss1+b*loss2
                val_loss2 += loss.item()*images.size(0)
        
            val_loss2 = val_loss2 / len(straight_loader2)
            val_losses2.append(val_loss2)
            mlflow.log_metric("eval_loss_straight", val_loss2, epoch)
    
        model.eval() 
        val_loss3 = 0.0
        with torch.no_grad():
            for data in curved_loader2:
                images = data
                outputs = model(images)[0]
                loss1 = criterion(outputs,images)
         
                loss2 = criterion2(outputs,images)
            
                loss3 = CustomLoss(outputs,images)
          
                loss= c*loss1 + a*loss1+b*loss2
                val_loss3 += loss.item()*images.size(0)
            
            val_loss3 = val_loss3 / len(curved_loader2)
            val_losses3.append(val_loss3)
           
            
  
 
plt.plot(epoch_list,train_loss_list,label='Training Loss')
plt.plot( epoch_list,val_losses, label='Validation Loss')
plt.plot( epoch_list,val_losses2, label='Validation Loss Straight')
plt.plot( epoch_list,val_losses3, label='Validation Loss Curved') 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.legend()
plt.show()


test_loss = 0.0
num_batches = 0
   
with torch.no_grad():
        for data in test_loader:
            images=data
            outputs = model(images)[0]
            outputs2=model(images)[1]
            loss1 = criterion(outputs,images)
            
            loss2 = criterion2(outputs,images)
            loss3 = CustomLoss(outputs,images)
                
            loss= c*loss1 + a*loss3+b*loss2
            test_loss += loss.item()
            
        
            num_batches += 1
    
        
        
        # Calculate average test loss
avg_test_loss = test_loss / num_batches 
print("Average Test Loss:", avg_test_loss)


print(images.size(),outputs.size(),outputs2.size())
print(images[0],outputs[0],outputs2[0])

n = 16  # Number of images
fig, axes = plt.subplots(3, n, figsize=(n + 10, 4))
    
for i in range(n):
        # Display original image
        ax = axes[0, i]
        ax.imshow(images[i].squeeze().detach().numpy(), cmap='gray')
        ax.set_title('Original', fontsize=10)
        ax.axis('off')
        
        
        # Display output image
        ax = axes[1, i]
        ax.imshow(outputs[i].squeeze().detach().numpy(), cmap='gray')
        ax.set_title('Output', fontsize=10)
        ax.axis('off')
        reshaped_tensor = outputs2.view(16, 1, 4, 3)
        ax = axes[2, i]
        ax.imshow(reshaped_tensor[i].squeeze().detach().numpy(), cmap='gray')
        ax.set_title('12-bit Output', fontsize=10)
        ax.axis('off')
        ax.text(0.5, -0.2, f'{i + 1}', ha='center', va='top', transform=ax.transAxes, fontsize=8)
        
        # Adding vertical lines to separate images
        for i in range(1, n):
            x = i / n
            line = Line2D([x, x], [0, 1], transform=fig.transFigure, color='black', linestyle='--', linewidth=1)
            fig.add_artist(line)

plt.tight_layout()
plt.show()


    #Testing 2
    
test_loss2 = 0.0
num_batches2 = 0
with torch.no_grad():
    for data in straight_loader:
            images=data
            outputs = model(images)[0]
            outputs2=model(images)[1]
            loss1 = criterion(outputs,images)
            
            loss2 = criterion2(outputs,images)
                
            loss3 = CustomLoss(outputs,images)
        
            loss= c*loss1 + a*loss3+b*loss2
            
            test_loss2 += loss.item()
            
            num_batches2 += 1
    
        

    # Calculate average test loss
avg_test_loss2 = test_loss2 / num_batches2
print("Average Test Loss:", avg_test_loss2)
   

test_loss3 = 0.0
num_batches3 = 0
with torch.no_grad():
        for data in curved_loader:
            images=data
            outputs = model(images)[0]
            outputs2 = model(images)[1]
            loss1 = criterion(outputs,images)
            
            loss2 = criterion2(outputs,images)
            
            loss3 = CustomLoss(outputs,images)
            
            loss= c*loss1 + a*loss3+b*loss2
            
            test_loss3 += loss.item()
            
            num_batches3 += 1
            
        

    # Calculate average test loss
avg_test_loss3 = test_loss3 / num_batches3
print("Average Test Loss:", avg_test_loss3)
  
name_mapping = {
        'encoder.0.weight': 'Encoder Conv Layer 1 Weights',
        'encoder.0.bias': 'Encoder Conv Layer 1 Biases',
        'encoder.3.weight': 'Encoder Conv Layer 2 Weights',
        'encoder.3.bias': 'Encoder Conv Layer 2 Biases',
        'encoder.7.weight': 'Encoder Linear 1 Weights',
        'encoder.7.bias': 'Encoder Linear Layer 1 Biases',
        'encoder.9.weight': 'Encoder Linear Layer 2 Weights',
        'encoder.9.bias': 'Encoder Linear Layer 2 Biases',
        'encoder.11.weight': 'Encoder Linear Layer 3 Weights',
        'encoder.11.bias': 'Encoder Linear Layer 3 Biases',
        'decoder.0.weight': 'Decoder ConvT Layer 1 Weights',
        'decoder.0.bias': 'Decoder ConvT Layer 1 Biases',
        'decoder.2.weight': 'Decoder ConvT Layer 2 Weights',
        'decoder.2.bias': 'Decoder ConvT Layer 2 Biases',
        'decoder.4.weight': 'Decoder Linear Layer 1 Weights',
        'decoder.4.bias': 'Decoder Linear Layer 1 Biases',
        'decoder.7.weight': 'Decoder Linear Layer 2 Weights',
        'decoder.7.bias': 'Decoder Linear Layer 2 Biases',
        'decoder.9.weight': 'Decoder Linear Layer 3 Weights',
        'decoder.9.bias': 'Decoder Linear Layer 3 Biases'
    }


    # Define colors for each layer
layer_colors = ['blue', 'orange', 'green', 'red','black','yellow','purple','lightgreen','grey','deepskyblue']

    # Extract weights and biases by layer with more descriptive names
    
weights_by_layer = []
biases_by_layer = []
weight_names = []
bias_names = []

for name, param in model.named_parameters():
        if 'weight' in name:
            weights_by_layer.append(param.data.numpy().flatten())
            weight_names.append(name_mapping[name])
        elif 'bias' in name:
            biases_by_layer.append(param.data.numpy().flatten())
            bias_names.append(name_mapping[name])
    # Plot histograms for weights
plt.figure(figsize=(12, 6))


    # Weights histogram
plt.subplot(1, 2, 1)
for i, (weights, name) in enumerate(zip(weights_by_layer, weight_names)):
    plt.hist(weights, bins=50, color=layer_colors[i % len(layer_colors)], alpha=0.5, label=name)
plt.title('Weights Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

    # Biases histogram
plt.subplot(1, 2, 2)
for i, (biases, name) in enumerate(zip(biases_by_layer, bias_names)):
    plt.hist(biases, bins=50, color=layer_colors[i % len(layer_colors)], alpha=0.5, label=name)
plt.title('Biases Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

    