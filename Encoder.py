#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:23:54 2024

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
mlflow.set_tracking_uri("http://0.0.0.0:5000")
   
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
batch_size = 32 #48# 32
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
class PixelWiseMSELoss(nn.Module):
    def __init__(self):
        super(PixelWiseMSELoss, self).__init__()

    def forward(self, input, target):
        assert input.shape == target.shape, "Input and target must have the same shape."
        
        # Apply sigmoid to ensure input values are in the range [0, 1]
        input = torch.sigmoid(input)
        
        # Calculate the BCE for each pixel
        pixel_bce = F.binary_cross_entropy(input, target, reduction='none')
        
        # Return the pixel-wise MSE as a tensor
        return pixel_bce
"""

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
           
            nn.Conv2d(1, 12, 3, padding=1),
            
     
            QuantReLU(bit_width=1),
            nn.MaxPool2d(2, 2),
        
     
            nn.Conv2d(12, 4, 3, padding=1),
            
            QuantReLU(bit_width=1),
            nn.MaxPool2d(2, 2))
       
        
        # Decoder layers
        self.decoder = nn.Sequential(
           
           
            nn.ConvTranspose2d(4, 12, 2, stride=2),
            QuantReLU(bit_width=8),
            #nn.ReLU(),
            nn.ConvTranspose2d(12, 1, 2, stride=2),
            
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        #encoded = torch.clamp(enc,min=None,max=1)
        
        #encoded = torch.clamp(enc,min=None,max=1)
       
        decoded = self.decoder(encoded)
        
        return decoded, encoded 
    
    
""" 
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder1 = nn.Sequential(
           
            nn.Conv2d(1, 12, 3, padding=1),
            
     
            QuantReLU(bit_width=1),
            nn.MaxPool2d(2, 2))
        self.encoder2= nn.Sequential(
     
            nn.Conv2d(12, 4, 3, padding=1),
            
            QuantReLU(bit_width=1),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
           
           
            nn.ConvTranspose2d(4, 12, 2, stride=2),
            QuantReLU(bit_width=8),
            #nn.ReLU(),
            nn.ConvTranspose2d(12, 1, 2, stride=2),
            nn.Sigmoid()
            #QuantReLU(bit_width=1)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder1(x)
        x = torch.clamp(x,min=None,max=1)
        x = self.encoder2(x)
        encoded = torch.clamp(x,min=None,max=1)
        #encoded = torch.clamp(enc,min=None,max=1)
       
        decoded = self.decoder(encoded)
        
        return decoded, encoded 


#Training
model = ConvAutoencoder()
#print(model)

# specify loss function
criterion = nn.MSELoss()
#nn.MSELoss( )
criterion2  = PixelWiseMSELoss()

a =0#0.0001#0.000001 #0.0000001
b = 0.18
c = 1-a-b
weight_decay=0.0001
lr=0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay) #0.00001
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
#scheduler = torch.ReduceLROnPlateau(optimizer, 'min')
# Extract weights and biases
# Create a dictionary mapping internal parameter names to descriptive names
name_mapping = {
    'encoder1.0.weight': 'Encoder Conv Layer 1 Weights',
    'encoder1.0.bias': 'Encoder Conv Layer 1 Biases',
    'encoder2.0.weight': 'Encoder Conv Layer 2 Weights',
    'encoder2.0.bias': 'Encoder Conv Layer 2 Biases',
    'decoder.0.weight': 'Decoder ConvT Layer 1 Weights',
    'decoder.0.bias': 'Decoder ConvT Layer 1 Biases',
    'decoder.2.weight': 'Decoder ConvT Layer 2 Weights',
    'decoder.2.bias': 'Decoder ConvT Layer 2 Biases'
}

# Define colors for each layer
layer_colors = ['blue', 'orange', 'green', 'red']

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

nweights = 0
for name,weights in model.named_parameters():
    if 'bias' not in name:
        nweights = nweights + weights.numel()
print(f'Total number of weights in the model = {nweights}')


n_epochs = 30
l1_lambda = 0.01#0.0001
losses = np.zeros(n_epochs+1)
#lan_l1=0.0001
#lan_l2=0
epoch_list=[]
train_loss_list=[]

val_losses=[]
val_losses2=[]
val_losses3=[]
with mlflow.start_run() as run:
    mlflow.log_param("conv1_kernel_size", 3)
    mlflow.log_param(   "conv1_stride", 1)
    mlflow.log_param(  "conv1_padding", 1)
    mlflow.log_param(   "conv2_kernel_size", 3)
    mlflow.log_param(   "conv2_stride", 1)
    mlflow.log_param(   "conv2_padding", 1)
    mlflow.log_param(  "pool_kernel_size",2)
    mlflow.log_param(  "pool_stride", 2)
    mlflow.log_param("tconv1_kernel_size", 2)
    mlflow.log_param(   "tconv1_stride", 2)
    mlflow.log_param(  "tconv1_padding",0)
    mlflow.log_param(   "tconv2_kernel_size", 2)
    mlflow.log_param(   "tconv2_stride", 2)
    mlflow.log_param(   "tconv2_padding", 0)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", n_epochs)
    mlflow.log_param("weight_decay", weight_decay)
    mlflow.log_param("L1_lamda", l1_lambda)
    mlflow.log_param("a", a)
    
    


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
            
            #loss3 = CustomLoss(outputs,images)
            
            loss= c*loss1 + b*loss2.mean()
            #c*loss1+ a*loss3
            
            
            L1_term = torch.tensor(0., requires_grad=True)
            for name, weights in model.named_parameters():
               if 'bias' not in name:
                    weights_sum = torch.sum(torch.abs(weights))
                    L1_term = L1_term + weights_sum
            L1_term = L1_term / nweights
           
            loss = loss- L1_term * l1_lambda
                        #losses[epoch] = loss.detach().item()
       
            optimizer.zero_grad()
            loss.backward()
    
            # perform a single optimization step (parameter update)
            optimizer.step()
   
            train_loss += loss.item()*images.size(0)
        
      
        
        
      
        train_loss = train_loss/len(train_loader)
        mlflow.log_metric("training_loss", train_loss , epoch)
       
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
                
                loss= c*loss1 + b*loss2.mean()
                val_loss += loss.item()*images.size(0)
                
            val_loss = val_loss / len(validation_loader)
            val_losses.append(val_loss)
            mlflow.log_metric("eval_loss", val_loss, epoch)
            #scheduler.step(val_loss)   
            model.eval() 
            val_loss2 = 0.0
        with torch.no_grad():
            for data in straight_loader2:
                images = data
                outputs = model(images)[0]
                loss1 = criterion(outputs,images)
                
                loss2 = criterion2(outputs,images)
                loss3 = CustomLoss(outputs,images)
         
                loss= c*loss1 + b*loss2.mean()
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
          
                loss= c*loss1 + b*loss2.mean()
                val_loss3 += loss.item()*images.size(0)
            
            val_loss3 = val_loss3 / len(curved_loader2)
            val_losses3.append(val_loss3)
            mlflow.log_metric("eval_loss_curved", val_loss3, epoch)
            
  
 
    plt.plot(epoch_list,train_loss_list,label='Training Loss')
    plt.plot( epoch_list,val_losses, label='Validation Loss')
    plt.plot( epoch_list,val_losses2, label='Validation Loss Straight')
    plt.plot( epoch_list,val_losses3, label='Validation Loss Curved')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves MSE')
    plt.legend()
    plt.show()


    test_loss = 0.0
    num_batches = 0
   # reconstruction_errors = []
    with torch.no_grad():
        for data in test_loader:
            images=data
            outputs = model(images)[0]
            outputs2=model(images)[1]
            loss1 = criterion(outputs,images)
            
            loss2 = criterion2(outputs,images)
            loss3 = CustomLoss(outputs,images)
                
            loss= c*loss1 + b*loss2.mean()
            test_loss += loss.item()
            
        
            num_batches += 1
    
        
        
        # Calculate average test loss
    avg_test_loss = test_loss / num_batches
    mlflow.log_metric("test_loss", avg_test_loss)
    print("Average Test Loss:", avg_test_loss)



#print(outputs.size())
#print(outputs2.size())
#print(outputs[0])
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
        
        ax = axes[2, i]
        ax.imshow(outputs2[i].squeeze().detach().numpy(), cmap='gray')
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
            
            #loss2 = criterion2(outputs,images)
                
            loss3 = CustomLoss(outputs,images)
        
            loss= loss1.mean()
            test_loss2 += loss.item()
            
            num_batches2 += 1
    
        

    # Calculate average test loss
    avg_test_loss2 = test_loss2 / num_batches2
    print("Average Test Loss:", avg_test_loss2)
    mlflow.log_metric("test_loss_straight", avg_test_loss2)


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
            
            loss= c*loss1 + b*loss2.mean()
            
            test_loss3 += loss.item()
            
            num_batches3 += 1
            
        

    # Calculate average test loss
    avg_test_loss3 = test_loss3 / num_batches3
    print("Average Test Loss:", avg_test_loss3)

    mlflow.log_metric("test_loss_curved",avg_test_loss3)
    
   # Save the model state_dict
    model_path = "model_state_dict.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path, "model")
    
    name_mapping = {
        'encoder1.0.weight': 'Encoder Conv Layer 1 Weights',
        'encoder1.0.bias': 'Encoder Conv Layer 1 Biases',
        'encoder2.0.weight': 'Encoder Conv Layer 2 Weights',
        'encoder2.0.bias': 'Encoder Conv Layer 2 Biases',
        'decoder.0.weight': 'Decoder ConvT Layer 1 Weights',
        'decoder.0.bias': 'Decoder ConvT Layer 1 Biases',
        'decoder.2.weight': 'Decoder ConvT Layer 2 Weights',
        'decoder.2.bias': 'Decoder ConvT Layer 2 Biases'
    }

    # Define colors for each layer
    layer_colors = ['blue', 'orange', 'green', 'red']

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

    