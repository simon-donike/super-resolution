#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


# imports
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

import wandb


# local imports
from models_parameters import losses
from models_parameters import models
from utils.dataloader_SRCNN import Dataset
from utils.prepare_dataset import prepare_dataset


# ## Create DataLoader Object

# In[2]:


# inputs
spot6_mosaic = '/home/simon/CDE_UBS/thesis/data_collection/spot6/spot6_mosaic.tif'
spot6_path = "/home/simon/CDE_UBS/thesis/data_collection/spot6/"
sen2_path = "/home/simon/CDE_UBS/thesis/data_collection/sen2/merged_reprojected/"
closest_dates_filepath = "/home/simon/CDE_UBS/thesis/data_loader/data/closest_dates.pkl"


# In[3]:


# get dataset object
dataset = Dataset(spot6_mosaic,sen2_path,spot6_path,closest_dates_filepath,window_size=500,factor=(10/1.5))
# create dataloader object
loader = DataLoader(dataset,batch_size=1, shuffle=True, num_workers=1)
print("Loader Length: ",len(loader))


# ## Prep Env and Load Model

# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[5]:


model = models.SRCNN()
loss_func = losses.loss_mse


# ## Train Model

# In[6]:


# implementation of model trainer function
def train_model(model,batch_size=1,lr=0.01,epochs=10,wandb_name="test"):
    
    logging=True
    if logging==True:
        wandb.init(project=wandb_name, entity="simon-donike")
        wandb.config = {
          "learning_rate": lr,
          "epochs": epochs,
          "batch_size": batch_size
        }
    
    # define loaders
    loader_train = DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=1)
    loader_test  = DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=1)
    loader_full  = DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=1)


    train_loss = []  # where we keep track of the training loss
    train_accuracy = []  # where we keep track of the training accuracy of the model
    val_loss = []  # where we keep track of the validation loss
    val_accuracy = []  # where we keep track of the validation accuracy of the model
    epochs = epochs  # number of epochs

    # initialize model

    model = model.double()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for e in range(epochs):
        model.train()
        train_correct = 0
        for (x_train_batch, y_train_batch) in loader_train:
            x_train_batch = x_train_batch.to(torch.double)
            x_train_batch = x_train_batch.to(device)

            y_train_batch = y_train_batch.to(torch.double) 
            y_train_batch = y_train_batch.to(device)
            y_hat = model(x_train_batch)  # forward pass

            loss = loss_func(y_hat, y_train_batch)  # compute the loss

            loss.backward()  # obtain the gradients with respect to the loss
            optimizer.step()  # perform one step of gradient descent
            optimizer.zero_grad()  # reset the gradients to 0
            y_hat_class = torch.argmax(y_hat.detach(), axis=1)  # we assign an appropriate label based on the network's prediction
            train_correct += torch.sum(y_hat_class == y_train_batch)
            train_loss.append(loss.item() / len(x_train_batch))
            if logging==True:
                wandb.log({'loss': loss.item() / len(x_train_batch)})

        train_accuracy.append(train_correct / len(loader_train.dataset))
        if logging==True:
            wandb.log({'train_acc': train_correct / len(loader_train.dataset)})


        print ('Epoch', e+1, ' finished.')

    if logging==True:
        wandb.finish()


# In[7]:


train_model(model=model,batch_size=1,lr=0.01,epochs=25,wandb_name="SRCNN")


# In[ ]:




