import torch
import cv2
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ImagenetLoaderValidation
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

#change these paramaeters to change the channel number and the activation threshold
ChannelNum=91
Thres=14

data_path='./imagenet/imagenet_tmp/raw_data/validation'

dataset_val= ImagenetLoaderValidation.ImageNetDataset(data_path, is_train = False)

BATCH_SIZE =16

data_loader_val = DataLoader(dataset_val, BATCH_SIZE, num_workers=8)



#resnet18 = models.resnet18(pretrained=True)
#alexnet = models.alexnet(pretrained=True)
#squeezenet = models.squeezenet1_0(pretrained=True)
"""
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        image_modules = list(models.resnet18().children())[:-1] #all layer expect last layer
        self.modelA = nn.Sequential(*image_modules)
        
    def forward(self, image):
        a = self.modelA(image)
        x = F.sigmoid(x)
        return x
"""


class MyModel(nn.Module):
    def __init__(self,L):
        super(MyModel, self).__init__()
        image_modules = list(models.vgg19_bn(pretrained=True).features)[:L] #first 22 layers
        self.modelA = nn.Sequential(*image_modules)
        
    def forward(self, image):
        x = self.modelA(image)
        return x

LayerLists=[]
image_modules=list(models.vgg19_bn(pretrained=True).features)
for ind,m in enumerate(image_modules):
        print(m)
        if isinstance(m, nn.BatchNorm2d):
                print(m)
                LayerLists.append(ind+1)
print(LayerLists)
LayerLists=[51]
minresponses=[]
maxresponses=[]
minindices=[]
maxindices=[]
minimgids=[]
maximgids=[]
for L in LayerLists:
    model = MyModel(L).cuda()
    model.eval()
    print("NEw Layer")
    GlobalCount=0
    with torch.no_grad():
        currentmin=0
        currentmax=0
        minindex=0
        maxindex=0
        minimg=0
        maximg=0
        for batch_id, batch in enumerate(data_loader_val):
                GlobalCount+=1
                data=batch[0].cuda()
                label=batch[1].cuda()
                
                respones=model(data)
                
                 
                respones=respones.permute(1,0,2,3)
                respones=respones.cpu().detach().numpy()
                
                if np.amax(respones[ChannelNum,:,:,:])>Thres:
                    
                    print(GlobalCount)
                    
                    for im in range(respones.shape[1]):
                        if np.amax(respones[ChannelNum,im,:,:])>Thres:
                            inimg=data.cpu().detach().numpy()[im,:,:,:]
                            inimg=np.transpose(inimg,[1,2,0])
                            inimg-=np.amin(inimg)
                            inimg=(inimg/np.amax(inimg))
                            print(inimg.shape)
                            print(np.amax(respones[ChannelNum,im,:,:]))
                            fig, ax = plt.subplots()
                            ax.imshow(inimg)
                            ax.axis('off')
                            plt.show()
                            arr=respones[ChannelNum,:,:,:].reshape(-1)
                            arr[0]=-15
                            sns.distplot(arr+0.5, hist = False, kde = True,kde_kws = {'linewidth': 3})
                            plt.grid()
                            plt.title('Probability Density Function of Activations', fontsize=26)
                            plt.ylabel('Probability Density', fontsize=20)
                            plt.xlabel('Activation Value', fontsize=20)
                            plt.show()           
              
                 
                print(np.amin(respones[ChannelNum,:,:,:]))
                
                """   
                #uncomment this part to display the activations in this kernel
                
                for i in range(respones.shape[0]):
                    arr=respones[i,:]
                    #arr=arr-np.mean(arr)
                    #arr=arr/np.var(arr)
                    #print(np.amin(arr))
                    if np.amax(respones[i,:])>Thres:
                                 print(i )
                                 sns.distplot(respones, hist = False, kde = True,kde_kws = {'linewidth': 3})
                                 plt.show()
                                 print(np.mean(arr))
                                 print(np.var(arr))
                                 sns.distplot(arr, hist = False, kde = True,kde_kws = {'linewidth': 3})
                                 plt.show()
                                
                #respones=respones-np.tile(np.expand_dims(np.mean(respones,1),-1),[1,respones.shape[1]])
                #respones=respones/np.tile(np.expand_dims(np.std(respones,1),-1),[1,respones.shape[1]])
                """
        print(GlobalCount)              
